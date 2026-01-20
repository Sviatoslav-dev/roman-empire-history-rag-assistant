from typing import List

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from logger import get_logger

logger = get_logger(__name__)

class WikipediaArticleFilter:
    """Applies quality filters to Wikipedia articles."""

    MIN_ARTICLE_LENGTH = 2000  # Minimum number of characters in the articles
    MIN_ARTICLE_CITATIONS = 3  # Minimum number of citations in the articles


    def filter_articles(self, articles: List[WikipediaArticleScraper]) -> List[WikipediaArticleScraper]:
        """Filter article scrapers using quality criteria."""
        filtered: List[WikipediaArticleScraper] = []
        for article in articles:
            try:
                if article.passes_quality_filters(self.MIN_ARTICLE_LENGTH, self.MIN_ARTICLE_CITATIONS):
                    filtered.append(article)
            except Exception as e:
                logger.error("Filter error for %s: %s", article.title, e)
        return filtered
