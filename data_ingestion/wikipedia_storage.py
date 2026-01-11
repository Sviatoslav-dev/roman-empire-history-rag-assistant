import os
from pathlib import Path
from typing import List
from urllib.parse import unquote

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from logger import get_logger

logger = get_logger(__name__)

ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "./data/articles"))

class WikipediaStorage:
    def get_downloaded_articles(self) -> List[WikipediaArticleScraper]:
        """
        Load saved Wikipedia article HTML files from `ARTICLES_DIR`
        and return scraper instances for them.
        """

        articles: List[WikipediaArticleScraper] = []

        if not ARTICLES_DIR.exists() or not ARTICLES_DIR.is_dir():
            logger.error("No valid articles directory provided.")
            return []

        for fp in sorted(ARTICLES_DIR.glob("*.html")):
            if not fp.exists() or not fp.is_file():
                logger.error("HTML file does not exist: %s", fp)
                return []

            try:
                html = fp.read_text(encoding="utf-8")
                title = fp.stem  # filename without extension
                articles.append(WikipediaArticleScraper(html, title))
            except Exception as e:
                logger.error("Error reading HTML file %s: %s", fp, e)
                return []

        return articles

    def article_exists(self, title: str) -> bool:
        safe_name = self.article_title_to_filename(title)
        file_path = ARTICLES_DIR / f"{safe_name}.html"
        return file_path.exists()

    def save_article_to_file(self, title: str, html: str) -> None:
        filename = self.article_title_to_filename(title)
        file_path = ARTICLES_DIR / f"{filename}.html"

        file_path.write_text(html, encoding="utf-8")

    def article_title_to_filename(self, title: str) -> str:
        """Return a filesystem-safe filename (without extension) for a title."""
        safe_name = unquote(title).replace(" ", "_")
        safe_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in safe_name)
        return safe_name
