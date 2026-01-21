from typing import List

from dotenv import load_dotenv

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.pg_metadata_store import get_pg_metadata_store
from data_ingestion.wikipedia_api_client import WikipediaApiClient
from logger import get_logger
from data_ingestion.chunk_models import ArticleChunk

load_dotenv()

logger = get_logger(__name__)
_meta = get_pg_metadata_store()
_wikipedia_client = WikipediaApiClient()

class WikipediaArticleFilter:
    """Applies quality filters to Wikipedia articles."""

    MIN_ARTICLE_LENGTH = 2000  # Minimum number of characters in the articles
    MIN_ARTICLE_CITATIONS = 3  # Minimum number of citations in the articles

    def filter_articles(self, articles: List[WikipediaArticleScraper]) -> List[WikipediaArticleScraper]:
        """Filter article scrapers using quality criteria.

        Quality metrics are expected to be precomputed and stored in Postgres
        (`ingestion_article`). This method reads metrics and filters without
        writing anything back.

        If metrics are missing for an article, we fall back to computing them
        from the HTML to avoid silently dropping content.
        """
        filtered: List[WikipediaArticleScraper] = []

        for article in articles:
            try:
                quality = _meta.get_article_quality(article.title)

                if quality is None:
                    # Backward-compatible fallback: compute from HTML if DB row is incomplete.
                    visible_text = article._get_visible_text()
                    symbols_number = len(visible_text)
                    citations_number = article.count_citations()
                    has_problem_or_update_box = article.has_problem_or_update_box()
                    is_english_available = article.is_english_article()
                else:
                    symbols_number = quality.symbols_number
                    citations_number = quality.citations_number
                    has_problem_or_update_box = quality.has_problem_or_update_box
                    is_english_available = quality.is_english_available

                if symbols_number <= self.MIN_ARTICLE_LENGTH:
                    continue
                if citations_number < self.MIN_ARTICLE_CITATIONS:
                    continue
                if has_problem_or_update_box:
                    continue
                if not is_english_available:
                    continue

                filtered.append(article)
            except Exception as e:
                logger.error("Filter error for %s: %s", getattr(article, "title", "<unknown>"), e)

        return filtered

    def is_license_allowed(self, license: str) -> bool:
        """Return True if the image license looks safe-to-use.

        We conservatively reject common non-free / fair-use indicators.
        When metadata is missing, we default to allowing.

        Args:
            license: License string (already extracted from metadata store).
        """

        tokens = set(license.split())

        forbidden_triggers: tuple[str, ...] = (
            # clearly non-free / restricted
            "fair use",
            "fair",
            "non free",
            "nonfree",
            "copyright",
            "all rights reserved",
            "noncommercial",
            "no derivatives",
            "nc",
            "nd",

            # ambiguous / not a specific reusable licence label (exclude by default)
            "attribution",
            "no restrictions",
            "copyrighted free use",

            # licences/labels that require extra obligations or are unclear for images (exclude by default)
            "gfdl",
            "lgpl",
            "fal",
        )

        for trigger in forbidden_triggers:
            if " " in trigger:
                if trigger in license:
                    return False
            else:
                if trigger in tokens:
                    return False

        return True


    def filter_chunk_images(self, chunks: List[ArticleChunk]) -> List[ArticleChunk]:
        # Keep the API pure: return new chunks with filtered images.
        # We avoid deepcopy (dataclasses include non-copyable fields sometimes).
        filtered: List[ArticleChunk] = []

        for chunk in chunks:
            kept_images = []
            for mention in chunk.images:
                license = _meta.get_image_by_url(mention.image.url).licence
                if license and self.is_license_allowed(license):
                    kept_images.append(mention)

            filtered.append(
                ArticleChunk(
                    page_title=chunk.page_title,
                    page_url=chunk.page_url,
                    section_title=chunk.section_title,
                    section_path=chunk.section_path,
                    section_level=chunk.section_level,
                    text_parts=list(chunk.text_parts),
                    images=kept_images,
                    text=chunk.text,
                )
            )

        return filtered
