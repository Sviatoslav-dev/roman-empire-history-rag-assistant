from typing import Optional

import requests

import wikipedia
from wikipedia import PageError, HTTPTimeoutError, WikipediaException

from logger import get_logger

logger = get_logger(__name__)


class WikipediaApiClient:
    """Client for fetching raw Wikipedia HTML pages."""

    BASE_URL = "https://en.wikipedia.org"
    BASE_API_URL = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "RomanEmpireResearchBot/1.0"}
    REQUEST_TIMEOUT = 10  # seconds

    def __init__(self) -> None:
        """Initialize the Wikipedia API client and set default request headers."""
        wikipedia.set_lang("en")

    def fetch_article(self, title) -> str | None:
        """
        Fetch the raw HTML of a Wikipedia article by its title.

        Resolves disambiguation pages by selecting the first available option.

        Args:
            title: Article title (page name) to fetch.

        Returns:
            The article HTML as a string, or None if fetching fails.
        """

        try:
            page = wikipedia.page(title, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            if e.options:
                logger.info("Disambiguation for '%s', selecting:  %s", title, e.options[0])
                return self.fetch_article(e.options[0])
            logger.warning("Disambiguation for '%s' but no options available", title)
            return None
        except (PageError, HTTPTimeoutError, WikipediaException) as e:
            logger.error("Error fetching page '%s': %s", title, e)
            return None

        try:
            html = page.html()
        except WikipediaException:
            logger.exception("Error getting HTML for '%s'", title)
            return None

        if not html:
            logger.warning("No HTML content for '%s'", title)
            return None
        return html

    def fetch_category(self, category_name: str) -> Optional[str]:
        """
        Fetch the raw HTML of a Wikipedia category page.

        Args:
            category_name: The category name (without the "Category:" prefix).

        Returns:
            HTML content of the category page as text or None on failure.
        """

        url = f"{self.BASE_URL}/wiki/Category:{category_name.replace(' ', '_')}"
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            logger.exception("Failed to fetch category page: %s", category_name)
            return None
