from __future__ import annotations

from data_ingestion.scraper.base_page_scraper import BasePageScraper
from logger import get_logger

logger = get_logger(__name__)


class WikipediaArticleScraper(BasePageScraper):
    """Parses and analyzes a single Wikipedia article HTML."""

    PROBLEM_KEYWORDS = frozenset([
        "disput", "cleanup", "update", "problem", "outdat",
        "bias", "needs sources", "merge", "contradictory"
    ])

    BANNER_SELECTORS = frozenset([
        ".ambox", ".hatnote", ".mw-warning", ".metadata", ".notice", ".messagebox"
    ])


    def passes_quality_filters(self, min_article_length: int, min_citations_number: int) -> bool:
        """
        Check whether the article meets basic quality requirements.

        The article is considered valid if it:
        - contains sufficient visible text (more than min_article_length characters),
        - has an adequate number of citations (min_citations_number or more),
        - does not include maintenance/problem/update banners,
        - represents an English Wikipedia article.

        Args:
            min_article_length: Minimum number of visible characters required.
            min_citations_number: Minimum number of unique citation anchors.

        Returns:
            True if the article passes all quality filters, False otherwise.
        """
        # Length check: use visible text inside content area
        visible_text = self._get_visible_text()
        if len(visible_text) < min_article_length:
            logger.info("Article %s failed filter: %s", self.title, f"too_short (length={len(visible_text)})")
            return False

        # Citations count
        citations = self.count_citations()
        if citations < min_citations_number:
            logger.info("Article %s failed filter: %s", self.title, f"few_citations (count={citations})")
            return False

        # Problem or update box
        if self.has_problem_or_update_box():
            logger.info("Article %s failed filter: %s", self.title, "has_problem_or_update_box")
            return False

        # English availability
        if not self.is_english_article():
            logger.info("Article %s failed filter: %s", self.title, "not_english")
            return False

        return True


    def count_citations(self) -> int:
        """Estimate the number of unique inline citations in the article."""
        refs = set()
        for a in self.soup.select("a[href^='#cite_note']"):
            refs.add(a["href"])
        return len(refs)

    def has_problem_or_update_box(self) -> bool:
        """Detect whether the page has maintenance/problem/update/dispute banners."""
        for sel in self.BANNER_SELECTORS:
            for el in self.soup.select(sel):
                text = el.get_text()
                normalized = ' '.join(text.split()).lower()
                if any(kw in normalized for kw in self.PROBLEM_KEYWORDS):
                    return True

        # Also check for templates rendered as tables with class 'ambox'
        for el in self.soup.select("table.ambox, div.ambox"):
            text = ' '.join(el.get_text().split()).lower()
            if any(keyword in text for keyword in self.PROBLEM_KEYWORDS):
                return True

        # Check top of content for maintenance phrases
        content = self.soup.select_one("#mw-content-text")
        if content:
            paragraphs = content.select("p")[:2]
            text_parts = [' '.join(p.get_text().split()) for p in paragraphs]
            top_text = ' '.join(text_parts).lower()

            if any(keyword in top_text for keyword in self.PROBLEM_KEYWORDS):
                return True

        return False

    def is_english_article(self) -> bool:
        """Check if the page HTML is English.

        Returns:
            True when the HTML language is English or the content looks like an
            English Wikipedia article; False otherwise.
        """
        html_tag = self.soup.find('html')
        if html_tag is not None:
            lang = html_tag.get('lang') or html_tag.get('xml:lang')
            if lang and lang.lower().startswith('en'):
                return True

        # If the HTML <html> tag is missing (we have an HTML fragment), look
        # for any element that explicitly declares a language attribute.
        for tag in self.soup.find_all(attrs={ 'lang': True }):
            if (tag.get('lang') or '').lower().startswith('en'):
                return True
        for tag in self.soup.find_all(attrs={ 'xml:lang': True }):
            if (tag.get('xml:lang') or '').lower().startswith('en'):
                return True

        # Fallback: if the document contains typical Wikipedia content containers
        # we assume it's an English Wikipedia snippet that should be processed.
        if self.soup.select_one('#mw-content-text') is not None or self.soup.select_one('.mw-parser-output') is not None:
            return True

        return False

    def _get_visible_text(self) -> str:
        content_el = self.soup.select_one("#mw-content-text")
        return " ".join(content_el.get_text().split()) if content_el else " ".join(self.soup.get_text().split())
