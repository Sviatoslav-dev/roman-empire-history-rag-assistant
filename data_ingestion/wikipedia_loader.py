import os
import time
from pathlib import Path
from typing import Set, List

from dotenv import load_dotenv
from tqdm import tqdm

from data_ingestion.scraper.wikipedia_category_scraper import WikipediaCategoryScraper
from data_ingestion.wikipedia_api_client import WikipediaApiClient
from data_ingestion.wikipedia_storage import WikipediaStorage
from logger import get_logger

load_dotenv()

logger = get_logger(__name__)

_wikipedia_client = WikipediaApiClient()

ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "./data/articles"))

class WikipediaLoader:
    """Coordinates loading Wikipedia data from external sources."""

    RATE_LIMIT_DELAY = 0.5

    def get_all_articles_from_categories(self, categories: List[str]) -> Set[str]:
        """Collect article titles from the given Wikipedia categories.

        Args:
            categories: List of category names to crawl.

        Returns:
            A set of article titles (strings) discovered in the categories.
        """
        articles = set()

        for category in categories:
            logger.info(f"Crawling category: {category}")

            html = _wikipedia_client.fetch_category(category)
            if not html:
                logger.warning(f"Failed to fetch articles from category: {category}; skipping")
                continue

            scraper = WikipediaCategoryScraper(html, category)
            articles.update(scraper.extract_articles_from_category())

        return articles

    def fetch_pages_by_titles(self, titles: Set[str]) -> List[str]:
        """Download multiple Wikipedia articles as HTML files into `ARTICLES_DIR`.

        Args:
            titles: A set of article titles to fetch.

        Returns:
            A list of downloaded page HTML contents (strings).
        """

        storage = WikipediaStorage()

        pages: List[str] = []
        for title in tqdm(titles, desc="Downloading Wikipedia articles"):
            if storage.article_exists(title):
                logger.info(f"Skipping {title} as it already exists.")
                continue

            page_html = _wikipedia_client.fetch_article(title)
            if page_html is None:
                logger.warning("Failed to fetch article: %s", title)
                time.sleep(self.RATE_LIMIT_DELAY)
                continue

            storage.save_article_to_file(title, page_html)
            pages.append(page_html)

            time.sleep(self.RATE_LIMIT_DELAY)  # polite rate limiting
        return pages

    def load_categories(self, categories_file: str) -> List[str]:
        """Load categories names from a newline-delimited text file.

        Args:
            categories_file: Path to a file containing category names, one per line.

        Returns:
            A list of category names (strings).
        """
        with open(categories_file, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
