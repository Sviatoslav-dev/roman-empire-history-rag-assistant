from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
from typing import List

from dotenv import load_dotenv

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.wikipedia_api_client import WikipediaApiClient
from data_ingestion.wikipedia_loader import WikipediaLoader
from data_ingestion.wikipedia_storage import WikipediaStorage
from logger import get_logger
from rag.retriever import QdrantRetriever

_wikipedia_client = WikipediaApiClient()

load_dotenv()

logger = get_logger(__name__)

ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "./data/articles"))
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "./data/images"))


class WikipediaCollector:
    """High-level orchestration of Wikipedia article collection workflow."""

    MIN_ARTICLE_LENGTH = 2000  # Minimum number of characters in the articles
    MIN_ARTICLE_CITATIONS_NUMBER = 3 # Minimum number of citations in the articles

    def __init__(self, wikipedia_loader: WikipediaLoader, wikipedia_storage: WikipediaStorage) -> None:
        self.loader = wikipedia_loader
        self.storage = wikipedia_storage

        self.retriever = QdrantRetriever()


    def filter_articles(self, articles: List[WikipediaArticleScraper]) -> List[WikipediaArticleScraper]:
        """Filter article scrapers using quality criteria.

        Args:
            articles: List of WikipediaArticleScraper instances to evaluate.

        Returns:
            A list of scrapers that passed the quality filters.
        """
        filtered: List[WikipediaArticleScraper] = []

        for article in articles:
            try:
                if article.passes_quality_filters(
                        self.MIN_ARTICLE_LENGTH,
                        self.MIN_ARTICLE_CITATIONS_NUMBER,
                ):
                    filtered.append(article)
            except Exception as e:
                logger.error("Filter error for %s: %s", article.title, e)
        return filtered

    def collect_articles(self, categories_file: str) -> List[WikipediaArticleScraper]:
        """Load category names from a file, download articles, and filter them.

        Args:
            categories_file: Path to newline-delimited category file.

        Returns:
            A list of WikipediaArticleScraper instances that passed filters.
        """
        categories = self.loader.load_categories(categories_file)

        article_titles = self.loader.get_all_articles_from_categories(categories)

        if categories:
            logger.info("Found %d articles in categories %s", len(article_titles), categories)
        else:
            logger.warning("No valid categories provided in %s.", categories_file)

        if not article_titles:
            logger.warning("No article titles discovered from categories; aborting fetch.")
            return []

        self.loader.fetch_pages_by_titles(article_titles)

        downloaded_articles = self.storage.get_downloaded_articles()
        filtered_articles = self.filter_articles(downloaded_articles)

        failed = [r for r in downloaded_articles if r not in filtered_articles]
        logger.info("Articles scanned: %d; passed: %d; failed: %d", len(downloaded_articles), len(filtered_articles), len(failed))
        # return filtered_articles

        chunks = self.split_articles_into_chunks(filtered_articles)
        logger.info("Total article chunks created: %d", len(chunks))
        self.loader.download_images([image for chunk in chunks for image in chunk["images"]])


        chunk_texts = [chunk["text"] for chunk in chunks]
        for chunk in chunks:
            del chunk["text"]

        self.retriever.add_text_chunks(
            chunk_texts,
            chunks,
            ids=list(range(len(chunks))),
        )

        image_paths: List[str] = []
        image_metadata: List[Dict] = []
        failed_downloads = 0
        images = 0
        image_id = 0

        for chunk_metadata in chunks:

            for img_url in chunk_metadata["images"]:
                if "Blank.png" in img_url:
                    continue

                if "/thumb/" in img_url:
                    img_url = self.loader._convert_thumbnail_to_fullsize(img_url)
                # Handle File: URLs - these are page URLs, not direct image URLs
                # We should skip these as they require API calls to get the actual image URL
                if "/wiki/File:" in img_url or "/wiki/Image:" in img_url or "/w/extensions/wikihiero" in img_url:
                    # These are page URLs, not direct image URLs
                    # Skip them for now (would need MediaWiki API to resolve)
                    continue

                # Validate URL before processing
                if not img_url or not isinstance(img_url, str):
                    continue
                # Check for malformed URLs (duplicate filenames in path)
                url_parts = img_url.split("/")
                if len(url_parts) >= 2:
                    last_two = url_parts[-2:]
                    # If last two parts are the same (except for query params), it's malformed
                    if last_two[0].split("?")[0] == last_two[1].split("?")[0] and last_two[0].split("?")[0]:
                        # Remove the duplicate
                        img_url = "/".join(url_parts[:-1])

                # Normalize the URL
                if img_url.startswith("//"):
                    img_url = "https:" + img_url
                elif img_url.startswith("/"):
                    img_url = "https://en.wikipedia.org" + img_url

                # Try to download the image (download_image handles URL conversion)
                images += 1
                if images % 100 == 0:
                    print(f"Downloading image #{images}")
                # continue

                image_title = self.loader._extract_image_filename(img_url)
                local_path = self.storage.image_filepath(image_title)

                if "Western_and_Eastern_Roman_Empires_476AD%283%29.svg" in img_url:
                    print(f"Debug: Downloaded image path: {local_path}")

                if not local_path:
                    failed_downloads += 1
                    # Skip if download failed
                    continue

                image_paths.append(local_path)
                image_metadata.append(
                    {
                        "page_title": chunk_metadata["page_title"],
                        "page_url": img_url,
                        "section_title": chunk_metadata["section_title"],
                        "section_path": chunk_metadata["section_path"],
                        "image_url": img_url,
                        # "caption": f"Image from {page['title']} – {section.get('title_path')}",
                    }
                )
                image_id += 1

        CHUNK_SIZE = 500
        START_ID = 1_000_000

        for i in range(0, len(image_paths), CHUNK_SIZE):
            butch_paths = image_paths[i:i + CHUNK_SIZE]
            butch_metadata = image_metadata[i:i + CHUNK_SIZE] if image_metadata else None

            chunk_ids = list(
                range(
                    START_ID + i,
                    START_ID + i + len(butch_paths)
                )
            )

            self.retriever.add_images(
                butch_paths,
                butch_metadata,
                ids=chunk_ids,
            )

        return filtered_articles


    def split_articles_into_chunks(self, articles: List[WikipediaArticleScraper]) -> List[Dict]:
        chunks = []
        for article in articles:
            sections = article.split_by_sections()
            for section in sections:
                chunk = {
                    "page_title": article.title,
                    "page_url": article.url,
                    "section_title": section.get("title"),
                    "section_path": section.get("title_path"),
                    "section_level": section.get("level"),
                    "text": section.get("text"),
                    "images": section.get("images", []),
                }
                chunks.append(chunk)
        return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wikipedia client helper: download and filter saved article HTML files.")
    parser.add_argument(
        "--categories-file",
        type=str,
        default=os.getenv("CATEGORIES_FILE", "categories.txt"),
        help="Path to a text file with Wikipedia categories (one per line)"
    )


    args = parser.parse_args()

    loader = WikipediaLoader()
    storage = WikipediaStorage()
    collector = WikipediaCollector(loader, storage)
    articles = collector.collect_articles(args.categories_file)

    if articles:
        logger.info("Articles that passed the filters:")
        for p in articles:
            logger.info("- %s", p.title)
