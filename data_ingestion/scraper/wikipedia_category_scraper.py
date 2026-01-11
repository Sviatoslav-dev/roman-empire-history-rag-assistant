from typing import Optional, Set

from data_ingestion.scraper.base_page_scraper import BasePageScraper
from data_ingestion.wikipedia_api_client import WikipediaApiClient
from urllib.parse import unquote


class WikipediaCategoryScraper(BasePageScraper):

    def extract_articles_from_category(self) -> Set[str]:
        """Extract article titles from a category page."""
        articles: Set[str] = set()
        for group in self.soup.select("#mw-pages .mw-category-group"):
            for link in group.select("a[href^='/wiki/']"):
                title = unquote(link["href"]).replace("/wiki/", "")
                if ":" not in title:
                    articles.add(title.replace("_", " "))
        return articles
