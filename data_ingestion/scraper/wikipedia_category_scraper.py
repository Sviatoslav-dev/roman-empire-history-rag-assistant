from __future__ import annotations

from typing import Set
from urllib.parse import unquote

from data_ingestion.scraper.base_page_scraper import BasePageScraper


class WikipediaCategoryScraper(BasePageScraper):
    """Parses and analyzes a single Wikipedia category page HTML."""

    def extract_articles_from_category(self) -> Set[str]:
        """Extract article titles from a category page.

        Returns:
            A set of article titles (strings) discovered on the category page.
        """
        articles: Set[str] = set()
        for group in self.soup.select("#mw-pages .mw-category-group"):
            for link in group.select("a[href^='/wiki/']"):
                url = link["href"]
                title = unquote(url).replace("/wiki/", "")
                if ":" not in title:
                    articles.add(url)
        return articles
