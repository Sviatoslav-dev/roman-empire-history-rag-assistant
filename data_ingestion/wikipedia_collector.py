from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import unquote

from dotenv import load_dotenv
from tqdm import tqdm

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.scraper.wikipedia_category_scraper import WikipediaCategoryScraper
from data_ingestion.wikipedia_api_client import WikipediaApiClient
from logger import get_logger

load_dotenv()

logger = get_logger(__name__)

ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "./data/articles"))



class WikipediaCollector:
    """
    High-level orchestration layer for collecting Wikipedia articles.

    Responsibilities:
    - collect article titles from Wikipedia categories,
    - download raw HTML pages for articles,
    - load saved article HTML files,
    - apply quality filters to downloaded articles.
    """

    def __init__(self) -> None:
        """Initialize Wikipedia client."""
        self.wikipedia_client = WikipediaApiClient()

    def get_all_articles_from_categories(self, categories: list[str]) -> Set[str]:
        """Collect article titles from the given Wikipedia categories."""
        articles = set()

        for category in categories:
            logger.info(f"Crawling category: {category}")

            # extract articles via helper
            scraper = WikipediaCategoryScraper.get_by_title(category)
            articles.update(scraper.extract_articles_from_category())

        return articles


    def fetch_page(self, title: str) -> Optional[Dict]:
        """
        Download the raw HTML for a Wikipedia article and save it to
        the folder specified by the environment variable `ARTICLES_DIR`.
        """
        ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

        safe_name = self._safe_filename(title)
        file_path = ARTICLES_DIR / f"{safe_name}.html"

        if file_path.exists():
            return {"title": title, "path": str(file_path), "downloaded": False}

        page_html = self.wikipedia_client.fetch_article(title)

        file_path.write_text(page_html, encoding="utf-8")

        return {"title": title, "path": str(file_path), "downloaded": True}


    def fetch_pages_by_titles(self, titles: List[str]) -> List[Dict]:
        """Download multiple Wikipedia articles as HTML files into `ARTICLES_DIR`."""

        results: List[Dict] = []
        for title in tqdm(titles, desc="Downloading Wikipedia articles"):
            res = self.fetch_page(title)
            if res:
                results.append(res)
            time.sleep(0.5)  # polite rate limiting
        return results

    def get_downloaded_articles(self) -> List[WikipediaArticleScraper]:
        """
        Load saved Wikipedia article HTML files from `ARTICLES_DIR`
        and return scraper instances for them.
        """

        articles: List[WikipediaArticleScraper] = []
        files: List[Path] = [ARTICLES_DIR]

        for c in files:
            if not c.exists() or not c.is_dir():
                continue
            for fp in sorted(c.glob("*.html")):
                articles.append(WikipediaArticleScraper.get_from_file(fp))
        return articles

    def filter_articles(self, articles: List[WikipediaArticleScraper]) -> List[WikipediaArticleScraper]:
        """Filter article scrapers using quality criteria."""
        return [article for article in articles if article.passes_quality_filters()]

    def _safe_filename(self, title: str) -> str:
        """Return a filesystem-safe filename (without extension) for a title."""
        safe_name = unquote(title).replace(" ", "_")
        safe_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in safe_name)
        return safe_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wikipedia client helper: download and filter saved article HTML files.")
    parser.add_argument(
        "--categories-file",
        type=str,
        default=os.getenv("CATEGORIES_FILE", "categories.txt"),
        help="Path to a text file with Wikipedia categories (one per line)"
    )
    # Download is enabled by default; provide a --no-download to turn it off
    parser.add_argument("--download", dest="download", action="store_true", default=True, help="Enable downloading (default: enabled).")
    parser.add_argument("--no-download", dest="download", action="store_false", help="Disable downloading.")
    # Filtering is enabled by default; provide a --no-filter to turn it off
    parser.add_argument("--filter", dest="do_filter", action="store_true", default=True, help="Enable filtering of saved HTML files (default: enabled).")
    parser.add_argument("--no-filter", dest="do_filter", action="store_false", help="Disable filtering.")

    args = parser.parse_args()

    collector = WikipediaCollector()

    # Optional downloading from categories
    if args.download and args.categories_file:
        with open(args.categories_file, encoding="utf-8") as f:
            categories = [line.strip() for line in f if line.strip()]

        if categories:
            articles = collector.get_all_articles_from_categories(categories)
            logger.info("Found %d articles in categories %s", len(articles), categories)
            downloaded_pages = collector.fetch_pages_by_titles(list(articles))
            logger.info("Downloaded %d pages.", len(downloaded_pages))
        else:
            logger.warning("No valid categories provided to --categories.")

    # Optional filtering of saved HTML files
    if args.do_filter:
        downloaded_articles = collector.get_downloaded_articles()
        passed = collector.filter_articles(downloaded_articles)
        failed = [r for r in downloaded_articles if r not in passed]
        logger.info("Articles scanned: %d; passed: %d; failed: %d", len(downloaded_articles), len(passed), len(failed))
        if passed:
            logger.info("Passed articles:")
            for p in passed:
                logger.info("- %s", p.title)

        if failed:
            logger.info("\nFailed articles (showing up to 10 with reasons):")
            for p in failed[:10]:
                logger.info("- %s", p.title)

    if not (args.download or args.do_filter):
        logger.warning("Both download and filter are disabled. Nothing to do.")
        parser.print_help()
