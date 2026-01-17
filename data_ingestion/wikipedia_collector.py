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
        self.loader.download_images([image["src"] for chunk in chunks for image in chunk["images"]])


        # Store text chunks with IDs and prepare metadata
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_metadata = []

        for i, chunk in enumerate(chunks):
            metadata = {
                "page_title": chunk["page_title"],
                "page_url": chunk["page_url"],
                "section_title": chunk["section_title"],
                "section_path": chunk["section_path"],
                "section_level": chunk["section_level"],
                # "image_urls": chunk["images"],  # Store image URLs for reference
            }
            chunk_metadata.append(metadata)

        # Add text chunks to vector DB
        self.retriever.add_text_chunks(
            chunk_texts,
            chunk_metadata,
            ids=list(range(len(chunks))),
        )

        # Process and store images with proper links to text chunks
        image_paths: List[str] = []
        image_metadata: List[Dict] = []
        failed_downloads = 0
        images = 0

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = chunk_idx  # The text chunk ID

            for img in chunk["images"]:
                img_url = img["src"]
                if "Blank.png" in img_url:
                    continue

                if "/thumb/" in img_url:
                    img_url = self.loader._convert_thumbnail_to_fullsize(img_url)

                # Handle File: URLs - these are page URLs, not direct image URLs
                if "/wiki/File:" in img_url or "/wiki/Image:" in img_url or "/w/extensions/wikihiero" in img_url:
                    continue

                # Validate URL before processing
                if not img_url or not isinstance(img_url, str):
                    continue

                # Check for malformed URLs (duplicate filenames in path)
                url_parts = img_url.split("/")
                if len(url_parts) >= 2:
                    last_two = url_parts[-2:]
                    if last_two[0].split("?")[0] == last_two[1].split("?")[0] and last_two[0].split("?")[0]:
                        img_url = "/".join(url_parts[:-1])

                # Normalize the URL
                if img_url.startswith("//"):
                    img_url = "https:" + img_url
                elif img_url.startswith("/"):
                    img_url = "https://en.wikipedia.org" + img_url

                # Try to download the image
                images += 1
                if images % 100 == 0:
                    print(f"Downloading image #{images}")

                image_title = self.loader._extract_image_filename(img_url)
                local_path = self.storage.image_filepath(image_title)

                if not local_path:
                    failed_downloads += 1
                    continue

                image_paths.append(str(local_path))
                # Store metadata with proper connection to text chunk
                image_metadata.append(
                    {
                        "page_title": chunk["page_title"],
                        "page_url": chunk["page_url"],
                        "local_path": str(local_path),
                        "section_title": chunk["section_title"],
                        "section_path": chunk["section_path"],
                        "section_level": chunk["section_level"],
                        "image_url": img_url,
                        # "caption": "",
                        "text_chunk_id": chunk_id,  # Link to parent text chunk
                    }
                )

        logger.info(f"Total images to store: {len(image_paths)}, Failed downloads: {failed_downloads}")

        # Store images in batches with proper IDs
        BATCH_SIZE = 500
        IMAGE_ID_START = 1_000_000

        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            batch_metadata = image_metadata[i:i + BATCH_SIZE]

            batch_ids = list(
                range(
                    IMAGE_ID_START + i,
                    IMAGE_ID_START + i + len(batch_paths)
                )
            )

            self.retriever.add_images(
                batch_paths,
                batch_metadata,
                ids=batch_ids,
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
