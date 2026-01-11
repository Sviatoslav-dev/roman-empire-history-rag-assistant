from pathlib import Path
from typing import Optional

from data_ingestion.scraper.base_page_scraper import BasePageScraper
from data_ingestion.wikipedia_api_client import WikipediaApiClient
from logger import get_logger

logger = get_logger(__name__)


class WikipediaArticleScraper(BasePageScraper):

    def passes_quality_filters(self) -> bool:
        """
        Check whether the article meets basic quality requirements.

        The article is considered valid if it:
        - contains sufficient visible text (more than 2000 characters),
        - has an adequate number of citations(3 and more),
        - does not include maintenance/problem/update banners,
        - represents an English Wikipedia article.

        Returns:
            bool: True if the article passes all quality filters,
                  False otherwise.
        """
        min_article_length = 2000  # characters
        min_citations_number = 3

        # Length check: use visible text inside content area
        content_el = self.soup.select_one("#mw-content-text")
        visible_text = (" ".join(content_el.get_text().split()) if content_el else " ".join(self.soup.get_text().split()))
        if len(visible_text) <= min_article_length:
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
        """
        Estimate the number of unique inline citations in the article.

        Counts unique citation anchors that reference Wikipedia `cite_note`
        entries (links with href starting with '#cite_note').
        """
        refs = set()
        for a in self.soup.select("a[href^='#cite_note']"):
            refs.add(a["href"])
        return len(refs)

    def has_problem_or_update_box(self) -> bool:
        """Detect whether the page has maintenance/problem/update/dispute banners.

        We look for known banner classes (ambox, hatnote, mw-warning) and check
        their text for keywords like 'disput', 'cleanup', 'update', 'problem', 'outdated'.
        """
        banner_selectors = [".ambox", ".hatnote", ".mw-warning", ".metadata", ".notice", ".messagebox"]
        keywords = ("disput", "cleanup", "update", "problem", "outdat", "bias", "needs sources", "merge", "contradictory")
        for sel in banner_selectors:
            for el in self.soup.select(sel):
                text = (' '.join(el.get_text().split()) or "").lower()
                for kw in keywords:
                    if kw in text:
                        return True
        # Also check for templates rendered as tables with class 'ambox'
        for el in self.soup.select("table.ambox, div.ambox"):
            if el and any(kw in (' '.join(el.get_text().split()) or "").lower() for kw in keywords):
                return True
        # Check top of content for maintenance phrases
        content = self.soup.select_one("#mw-content-text")
        if content:
            top_text = ' '.join(' '.join(p.get_text().split()) for p in content.select('p')[:2]).lower()
            for kw in keywords:
                if kw in top_text:
                    return True
        return False

    def is_english_article(self) -> bool:
        """Check if the page HTML is English.

        We prefer the <html lang="en"> attribute. If absent (HTML fragment),
        check for any element with a lang or xml:lang attribute starting with 'en'.
        Finally, fall back to presence of common Wikipedia content containers
        (e.g. #mw-content-text or .mw-parser-output) which usually indicate an
        English Wikipedia page in this project.
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
