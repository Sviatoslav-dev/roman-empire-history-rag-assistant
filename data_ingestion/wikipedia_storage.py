import hashlib
import os
from pathlib import Path
from typing import List
from urllib.parse import unquote

from data_ingestion.pg_metadata_store import get_pg_metadata_store
from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.wikipedia_image import WikipediaImage
from logger import get_logger

logger = get_logger(__name__)

ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "./data/articles"))
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "./data/images"))

class WikipediaStorage:
    """Handles persistence of Wikipedia articles and related assets."""

    def get_downloaded_articles(self) -> List[WikipediaArticleScraper]:
        """Load saved Wikipedia article HTML files referenced by PostgreSQL.

        Reads rows from `ingestion_article` table (title/url/local_path), loads
        HTML from `local_path` on disk, and returns scraper instances.

        Returns:
            A list of `WikipediaArticleScraper` instances constructed from saved
            HTML files.
        """

        postgres = get_pg_metadata_store()
        if not postgres.enabled:
            logger.error("PostgreSQL metadata store is disabled; cannot load downloaded articles.")
            return []

        articles: List[WikipediaArticleScraper] = []
        rows = postgres.list_articles_with_local_path()
        if not rows:
            logger.warning("No ingestion articles with local_path found in PostgreSQL.")
            return []

        for row in rows:
            fp = Path(row.local_path)
            if not fp.exists() or not fp.is_file():
                logger.warning("Article HTML file missing on disk (title=%s): %s", row.title, fp)
                continue

            try:
                html = fp.read_text(encoding="utf-8")
            except Exception:
                logger.exception("Error reading article HTML file (title=%s): %s", row.title, fp)
                continue

            try:
                scraper = WikipediaArticleScraper(html, row.title, row.url)
                # Keep URL best-effort: WikipediaArticleScraper doesn't currently
                # accept url in __init__, so we attach it if possible.
                articles.append(scraper)
            except Exception:
                logger.exception("Failed to create WikipediaArticleScraper for title=%s", row.title)
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

    def image_filepath(self, title: str) -> Path:
        """Return the on-disk path where an image with the given filename should live."""
        filename = self.image_title_to_filename(title)
        filepath = IMAGES_DIR / filename
        return filepath

    def image_exists(self, title: str) -> bool:
        """Return True if an image file already exists on disk."""
        return self.image_filepath(title).exists()

    def article_title_to_filename(self, title: str) -> str:
        """Return a filesystem-safe filename (without extension) for a title."""
        safe_name = unquote(title).replace(" ", "_")
        safe_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in safe_name)
        return safe_name

    def image_title_to_filename(self, title: str) -> str:
        """Return a filesystem-safe filename (without extension) for an image title."""
        max_length = 255

        safe_name = unquote(title).replace(" ", "_")
        safe_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in safe_name)
        if max_length and len(safe_name) > max_length:
            name, ext = os.path.splitext(safe_name)
            h = hashlib.md5(safe_name.encode("utf-8")).hexdigest()[:8]
            cut_len = max_length - len(ext) - len(h) - 1  # 1 for "_"
            safe_name = f"{name[:cut_len]}_{h}{ext}"
        return safe_name

    def _extract_image_filename(self, image_url: str) -> str:
        return WikipediaImage(image_url).get_filename()
