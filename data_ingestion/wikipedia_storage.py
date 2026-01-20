from typing import List
from urllib.parse import unquote

from config import ARTICLES_DIR
from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from logger import get_logger

logger = get_logger(__name__)


class WikipediaStorage:
    """Handles persistence of Wikipedia articles and related assets."""

    def get_downloaded_articles(self) -> List[WikipediaArticleScraper]:
        """
        Load saved Wikipedia article HTML files from `ARTICLES_DIR`
        and return scraper instances for them.

        Returns:
            A list of `WikipediaArticleScraper` instances constructed from saved
            HTML files found in the `ARTICLES_DIR` directory.
        """

        articles: List[WikipediaArticleScraper] = []

        if not ARTICLES_DIR.exists() or not ARTICLES_DIR.is_dir():
            logger.error("No valid articles directory provided.")
            return []

        for fp in sorted(ARTICLES_DIR.glob("*.html")):
            if not fp.exists() or not fp.is_file():
                logger.warning("Skipping non-file entry in articles dir: %s", fp)
                continue

            try:
                html = fp.read_text(encoding="utf-8")
                title = fp.stem  # filename without extension
                try:
                    articles.append(WikipediaArticleScraper(html, title))
                except Exception:
                    logger.exception("Failed to construct scraper for file %s; skipping", fp)
                    continue
            except Exception:
                logger.exception("Error reading HTML file %s; skipping", fp)
                continue

        return articles

    def article_exists(self, title: str) -> bool:
        """Return True if an article file for the given title exists on disk."""
        safe_name = self.article_title_to_filename(title)
        file_path = ARTICLES_DIR / f"{safe_name}.html"
        return file_path.exists()

    def save_article_to_file(self, title: str, html: str) -> None:
        """Persist article HTML to the articles directory using a safe filename.

        Args:
            title: Article title used to generate the filename.
            html: Raw HTML content to write to disk.
        """
        ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

        filename = self.article_title_to_filename(title)
        file_path = ARTICLES_DIR / f"{filename}.html"

        file_path.write_text(html, encoding="utf-8")

    def article_title_to_filename(self, title: str) -> str:
        """Return a filesystem-safe filename (without extension) for a title."""
        safe_name = unquote(title).replace(" ", "_")
        safe_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in safe_name)
        return safe_name
