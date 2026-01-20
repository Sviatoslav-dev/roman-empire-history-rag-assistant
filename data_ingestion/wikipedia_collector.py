from __future__ import annotations

import os

from typing import List

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.wikipedia_article_filter import WikipediaArticleFilter
from data_ingestion.wikipedia_loader import WikipediaLoader
from data_ingestion.wikipedia_storage import WikipediaStorage
from logger import get_logger

logger = get_logger(__name__)

_article_filter = WikipediaArticleFilter()


class WikipediaCollector:
    """High-level orchestration of Wikipedia article collection workflow."""

    def __init__(self, wikipedia_loader: WikipediaLoader, wikipedia_storage: WikipediaStorage) -> None:
        self.loader = wikipedia_loader
        self.storage = wikipedia_storage


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
        filtered_articles = _article_filter.filter_articles(downloaded_articles)

        failed = [r for r in downloaded_articles if r not in filtered_articles]
        logger.info("Articles scanned: %d; passed: %d; failed: %d", len(downloaded_articles), len(filtered_articles), len(failed))
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
