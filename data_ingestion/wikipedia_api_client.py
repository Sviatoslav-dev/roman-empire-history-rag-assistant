import requests

import wikipedia

from logger import get_logger

logger = get_logger(__name__)


class WikipediaApiClient:
    """Client for fetching raw Wikipedia HTML pages."""

    BASE_URL = "https://en.wikipedia.org"
    BASE_API_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self):
        """Initialize the Wikipedia API client and set default request headers."""
        self.headers = {
            "User-Agent": "RomanEmpireResearchBot/1.0"
        }
        wikipedia.set_lang("en")

    def fetch_article(self, title):
        """
        Fetch the raw HTML of a Wikipedia article by its title.

        Resolves disambiguation pages by selecting the first available option.

        Returns:
            str | None: HTML content of the article, or None if fetching fails.
        """

        try:
            page = wikipedia.page(title, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            if e.options:
                return self.fetch_article(e.options[0])
            return None
        except Exception as e:
            logger.error("Error fetching page '%s': %s", title, e)
            return None

        try:
            html = page.html()
        except Exception as e:
            logger.error("Error getting HTML for '%s': %s", title, e)
            return None

        if not html:
            logger.warning("No HTML content for '%s'", title)
            return None
        return html

    def fetch_category(self, category_name):
        """
        Fetch the raw HTML of a Wikipedia category page.

        Returns:
            str: HTML content of the category page.

        Raises:
            requests.HTTPError: If the HTTP request fails.
        """

        url = f"{self.BASE_URL}/wiki/Category:{category_name.replace(' ', '_')}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.text
