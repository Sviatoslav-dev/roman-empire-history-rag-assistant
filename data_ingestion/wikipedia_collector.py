from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
from typing import List

from dotenv import load_dotenv

from data_ingestion.pg_metadata_store import get_pg_metadata_store
from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.wikipedia_api_client import WikipediaApiClient
from data_ingestion.wikipedia_article_filter import WikipediaArticleFilter
from data_ingestion.wikipedia_loader import WikipediaLoader
from data_ingestion.wikipedia_storage import WikipediaStorage
from data_ingestion.images_preprocessor import ImagesPreprocessor
from logger import get_logger
from rag.retriever import QdrantRetriever

_wikipedia_client = WikipediaApiClient()

load_dotenv()

logger = get_logger(__name__)
_meta = get_pg_metadata_store()

ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "./data/articles"))
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "./data/images"))

_article_filter = WikipediaArticleFilter()


class WikipediaCollector:
    """High-level orchestration of Wikipedia article collection workflow."""


    def __init__(self, wikipedia_loader: WikipediaLoader, wikipedia_storage: WikipediaStorage) -> None:
        self.loader = wikipedia_loader
        self.storage = wikipedia_storage

        self.retriever = QdrantRetriever()



    def collect_articles(self, categories_file: str) -> List[WikipediaArticleScraper]:
        """Load category names from a file, download articles, and filter them.

        Args:
            categories_file: Path to newline-delimited category file.

        Returns:
            A list of WikipediaArticleScraper instances that passed filters.
        """
        categories = self.loader.load_categories(categories_file)

        article_urls = self.loader.get_all_articles_from_categories(categories)

        if categories:
            logger.info("Found %d articles in categories %s", len(article_urls), categories)
        else:
            logger.warning("No valid categories provided in %s.", categories_file)

        if not article_urls:
            logger.warning("No article titles discovered from categories; aborting fetch.")
            return []

        self.loader.fetch_pages_by_titles(article_urls)

        downloaded_articles = self.storage.get_downloaded_articles()
        filtered_articles = _article_filter.filter_articles(downloaded_articles)

        failed = [r for r in downloaded_articles if r not in filtered_articles]
        logger.info("Articles scanned: %d; passed: %d; failed: %d", len(downloaded_articles), len(filtered_articles), len(failed))
        # return filtered_articles

        chunks = self.split_articles_into_chunks(filtered_articles)
        logger.info("Total article chunks created: %d", len(chunks))
        images = [image["image"] for chunk in chunks for image in chunk["images"]]
        self.loader.download_images(images)

        chunks = _article_filter.filter_chunk_images(chunks)

        # Post-process downloaded images: convert SVGs to PNG and remove broken raster files.
        # Kept here (right after downloads) so local files are ready before we store them in Qdrant.
        ImagesPreprocessor().convert_svgs_to_png()

        # --- Build text chunk payloads + image dedup maps ---
        # Normalized model:
        # - TEXT points contain ONLY text/section/page metadata (no image urls / local paths)
        # - IMAGE points contain ONLY unique image metadata + embedding
        # - LINK points contain the many-to-many relations + per-chunk caption/section context
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_metadata: List[Dict] = []

        unique_images: Dict[str, Dict] = {}
        links: List[Dict] = []

        # We'll assign deterministic ids for images we upsert in this run
        IMAGE_ID_START = 1_000_000
        next_image_id = IMAGE_ID_START

        for chunk_id, chunk in enumerate(chunks):
            metadata = {
                "page_title": chunk["page_title"],
                "page_url": chunk["page_url"],
                "section_title": chunk["section_title"],
                "section_path": chunk["section_path"],
                "section_level": chunk["section_level"],
            }
            chunk_metadata.append(metadata)

            for img in chunk.get("images", []) or []:
                # img = img["image"]
                img_url = img["image"].url
                if "Blank.png" in img_url:
                    continue

                # Convert/normalize to a direct absolute URL for better dedup + retrieval

                image_key = img_url
                # Create unique image record if not exists
                if img_url not in unique_images:
                    # image_title = img["image"].get_filename()
                    # local_path = self.storage.image_filepath(image_title)
                    local_path = _meta.get_image_by_url(img_url).local_path

                    unique_images[image_key] = {
                        "image_id": next_image_id,
                        "image_url": img_url,
                        "local_path": str(local_path),
                    }
                    next_image_id += 1

                image_id = unique_images[image_key]["image_id"]
                caption = (img.get("caption") or "").strip()

                links.append(
                    {
                        "text_chunk_id": chunk_id,
                        "image_id": image_id,
                        "caption": caption,
                        "page_title": chunk["page_title"],
                        "page_url": chunk["page_url"],
                        "section_title": chunk.get("section_title"),
                        "section_path": chunk.get("section_path"),
                        "section_level": chunk.get("section_level"),
                    }
                )

        # Add text chunks to vector DB
        self.retriever.add_text_chunks(
            chunk_texts,
            chunk_metadata,
            ids=list(range(len(chunks))),
        )

        # --- Store unique images in image collection ---
        image_paths: List[str] = []
        image_metadata: List[Dict] = []
        image_ids: List[int] = []

        for rec in unique_images.values():
            image_ids.append(rec["image_id"])
            image_paths.append(rec["local_path"])
            image_metadata.append(
                {
                    "image_url": rec["image_url"],
                    "local_path": rec["local_path"],
                }
            )

        logger.info(
            "Unique images to store: %d (from %d total image mentions)",
            len(image_paths),
            sum(len(chunk.get("images", []) or []) for chunk in chunks),
        )

        # Store images in batches
        BATCH_SIZE = 500
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            batch_metadata = image_metadata[i:i + BATCH_SIZE]
            batch_ids = image_ids[i:i + BATCH_SIZE]

            self.retriever.add_images(
                batch_paths,
                batch_metadata,
                ids=batch_ids,
            )

        # --- Store links in link collection ---
        # Make link ids stable for this ingestion run
        self.retriever.add_chunk_image_links(
            links,
            ids=list(range(0, len(links))),
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
