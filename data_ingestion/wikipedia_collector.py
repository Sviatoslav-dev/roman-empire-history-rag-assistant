from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

from data_ingestion.chunk_ingestion import ChunkIngestionPipeline
from data_ingestion.pg_metadata_store import get_pg_metadata_store
from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.wikipedia_article_filter import WikipediaArticleFilter
from data_ingestion.wikipedia_loader import WikipediaLoader
from data_ingestion.wikipedia_storage import WikipediaStorage
from logger import get_logger
from rag.retriever import QdrantRetriever

load_dotenv()

logger = get_logger(__name__)
_postgres = get_pg_metadata_store()
_article_filter = WikipediaArticleFilter()


class WikipediaCollector:
    """High-level orchestration of Wikipedia article collection workflow.

    This class is responsible for fetching and filtering Wikipedia articles.

    Chunk/image processing and payload preparation are delegated to
    `ChunkIngestionPipeline`.
    """

    def __init__(self, wikipedia_loader: WikipediaLoader, wikipedia_storage: WikipediaStorage) -> None:
        self.loader = wikipedia_loader
        self.storage = wikipedia_storage
        self.retriever = QdrantRetriever()

        self.chunk_pipeline = ChunkIngestionPipeline(
            loader=self.loader,
            article_filter=_article_filter,
            metadata_store=_postgres,
        )

    def collect_articles(self, categories_file: str) -> List[WikipediaArticleScraper]:
        """Collect, process, and ingest Wikipedia articles discovered from categories."""
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
        logger.info(
            "Articles scanned: %d; passed: %d; failed: %d",
            len(downloaded_articles),
            len(filtered_articles),
            len(failed),
        )

        self.chunk_pipeline.split_articles_into_chunks(filtered_articles)

        # Delegate chunk/image preparation
        self.chunk_pipeline.download_images()
        self.chunk_pipeline.postprocess_images()
        self.chunk_pipeline.filter_images_by_license()

        chunk_texts, chunk_metadata = self.chunk_pipeline.prepare_text_collection()
        unique_images_by_url, image_paths, image_metadata, image_ids = self.chunk_pipeline.prepare_images_collection()
        links = self.chunk_pipeline.prepare_link_collection(unique_images_by_url)

        # Upsert text chunks
        self.retriever.add_text_chunks(
            chunk_texts,
            chunk_metadata,
            ids=list(range(len(chunk_texts))),
        )

        # Upsert images
        if image_paths:
            self.retriever.add_images(image_paths, image_metadata, ids=image_ids)

        # Upsert links
        if links:
            self.retriever.add_chunk_image_links(links, ids=list(range(0, len(links))))

        return filtered_articles


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
