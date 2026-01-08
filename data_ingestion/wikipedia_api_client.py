import requests

import wikipedia

from logger import get_logger

logger = get_logger(__name__)


class WikipediaApiClient:
    BASE_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self):
        self.headers = {
            "User-Agent": "RomanEmpireResearchBot/1.0 (contact: your-email@example.com)"
        }
        wikipedia.set_lang("en")

    def fetch_article(self, title):
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
        return None

    def fetch_category(self, category_name):
        url = f"{self.BASE_URL}/wiki/Category:{category_name.replace(' ', '_')}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.text
