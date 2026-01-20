from typing import List

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.pg_metadata_store import get_pg_metadata_store
from logger import get_logger

logger = get_logger(__name__)
_meta = get_pg_metadata_store()

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
