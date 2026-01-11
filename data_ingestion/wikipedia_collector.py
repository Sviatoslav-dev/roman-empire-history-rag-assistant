from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.wikipedia_loader import WikipediaLoader
from data_ingestion.wikipedia_storage import WikipediaStorage
from logger import get_logger

load_dotenv()

logger = get_logger(__name__)

ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "./data/articles"))


class WikipediaCollector:
    """High-level orchestration of Wikipedia article collection workflow."""

    MIN_ARTICLE_LENGTH = 2000  # Minimum number of characters in the articles
    MIN_ARTICLE_CITATIONS_NUMBER = 3 # Minimum number of citations in the articles

    def __init__(self, wikipedia_loader: WikipediaLoader, wikipedia_storage: WikipediaStorage) -> None:
        self.loader = wikipedia_loader
        self.storage = wikipedia_storage


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
