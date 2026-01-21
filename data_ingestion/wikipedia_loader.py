import os
import time
from pathlib import Path
from typing import Set, List
from urllib.parse import unquote

from dotenv import load_dotenv
from tqdm import tqdm

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.scraper.wikipedia_category_scraper import WikipediaCategoryScraper
from data_ingestion.wikipedia_api_client import WikipediaApiClient
from data_ingestion.wikipedia_storage import WikipediaStorage
from data_ingestion.wikipedia_image import WikipediaImage
from data_ingestion.pg_metadata_store import get_pg_metadata_store, ArticleQuality
from logger import get_logger

load_dotenv()

logger = get_logger(__name__)

_wikipedia_client = WikipediaApiClient()
_storage = WikipediaStorage()
_meta = get_pg_metadata_store()

ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "./data/articles"))
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "./data/images"))

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
            discovered = scraper.extract_articles_from_category()

            # Progressive persistence: create DB records as soon as titles are discovered.
            for url in discovered:
                title = None
                try:
                    title = unquote(url).replace("/wiki/", "")
                    _meta.upsert_article_title(title)
                    _meta.update_article_url(title, url)
                    articles.add(url)
                except Exception:
                    logger.exception("Failed to upsert article title into PostgreSQL: %s", title or url)

        return articles

    def fetch_pages_by_titles(self, urls: Set[str]) -> List[str]:
        """Download multiple Wikipedia articles as HTML files into `ARTICLES_DIR`.

        Args:
            urls: A set of Wikipedia article URLs ("/wiki/..." form).

        Returns:
            A list of downloaded page HTML contents (strings).
        """

        pages: List[str] = []
        for url in tqdm(urls, desc="Downloading Wikipedia articles"):
            title = unquote(url).replace("/wiki/", "")

            if _storage.article_exists(title):
                logger.info(f"Skipping {title} as it already exists.")
                continue

            page_html = _wikipedia_client.fetch_article(title)
            if page_html is None:
                logger.warning("Failed to fetch article: %s", title)
                time.sleep(self.RATE_LIMIT_DELAY)
                continue

            _storage.save_article_to_file(title, page_html)

            try:
                file_path = ARTICLES_DIR / f"{_storage.article_title_to_filename(title)}.html"
                _meta.update_article_local_path(title, str(file_path))
            except Exception:
                logger.exception("Failed to update article local_path in PostgreSQL: %s", title)

            article = WikipediaArticleScraper(page_html, title, url)

            visible_text = article._get_visible_text()
            quality = ArticleQuality(
                symbols_number=len(visible_text),
                citations_number=article.count_citations(),
                has_problem_or_update_box=article.has_problem_or_update_box(),
                is_english_available=article.is_english_article(),
            )
            try:
                _meta.update_article_quality(title, quality)
            except Exception:
                logger.exception("Failed to persist article quality metrics: %s", article.title)

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


    def download_images(self, images: List[WikipediaImage]):
        """Download images and persist metadata incrementally."""
        for image in tqdm(images, desc="Downloading images"):
            logger.debug("Processing image URL: %s", image.url)

            image.normalize_url()

            # We key images by URL in Postgres.
            try:
                _meta.upsert_image_url(image.url)
            except Exception:
                logger.exception("Failed to upsert image URL into PostgreSQL: %s", image.url)

            image_title = image.get_filename()

            if _storage.image_exists(image_title):
                logger.info("Image already exists locally, skipping: %s", image_title)
                continue

            filepath = _storage.image_filepath(image_title)


            _wikipedia_client.download_image(image.url, filepath)
            image.local_path = filepath

            extmetadata = _wikipedia_client.get_image_license(image_title)
            licence_name = None
            if extmetadata:
                licence_name = (extmetadata.get("LicenseShortName") or {}).get("value")

            try:
                _meta.update_image_metadata(
                    url=image.url,
                    local_path=str(filepath),
                    filename=filepath.name,
                    extension=filepath.suffix.lstrip(".") or None,
                    licence=licence_name
                )
            except Exception:
                logger.exception("Failed to update downloaded image metadata in PostgreSQL: %s", image.url)

            time.sleep(0.5)
