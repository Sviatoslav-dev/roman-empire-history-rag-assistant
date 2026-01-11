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

    def get_all_articles_from_categories(self, categories: List[str]) -> Set[str]:
        """Collect article titles from the given Wikipedia categories."""
        articles = set()

        for category in categories:
            logger.info(f"Crawling category: {category}")

            html = _wikipedia_client.fetch_category(category)
            if not html:
                logger.info(f"Failed to fetch articles from category: {category}")
                return set()

            scraper = WikipediaCategoryScraper(html, category)
            articles.update(scraper.extract_articles_from_category())

        return articles

    def fetch_pages_by_titles(self, titles: Set[str]):
        """Download multiple Wikipedia articles as HTML files into `ARTICLES_DIR`."""

        ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

        for title in tqdm(titles, desc="Downloading Wikipedia articles"):
            storage = WikipediaStorage()
            if storage.article_exists(title):
                logger.info(f"Skipping {title} as it already exists.")
                continue

            page_html = _wikipedia_client.fetch_article(title)

            storage.save_article_to_file(title, page_html)

            time.sleep(0.5)  # polite rate limiting

    def load_categories(self, categories_file: str):
        with open(categories_file, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
