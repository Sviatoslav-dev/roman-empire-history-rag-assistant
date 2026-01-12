import time
from pathlib import Path

import requests

import wikipedia
from bs4 import BeautifulSoup

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

        # try:
        #     html = page.html()
        # except Exception as e:
        #     logger.error("Error getting HTML for '%s': %s", title, e)
        #     return None

        response = requests.get(page.url, headers=self.headers)
        response.raise_for_status()
        html = response.text


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

    def get_image_license(self, filename: str) -> dict | None:
        params = {
            "action": "query",
            "format": "json",
            "titles": f"File:{filename}",
            "prop": "imageinfo",
            "iiprop": "extmetadata"
        }

        r = requests.get(self.BASE_API_URL, params=params, headers=self.headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), None)

        if not page or "imageinfo" not in page:
            return None

        return page["imageinfo"][0]["extmetadata"]

    def download_image(self, image_url: str, filepath: Path) -> str | requests.Response:
        response = requests.get(image_url, headers=self.headers, timeout=30)

        if response.status_code == 429:
            time.sleep(60)
            response = requests.get(image_url, headers=self.headers, timeout=30)
            response.raise_for_status()

        if response.status_code == 200:
            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            content_length = response.headers.get("content-length", "0")
            # Accept if it's an image or has content
            if "image" in content_type or (content_length and int(content_length) > 0):
                # Write the image using streaming
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return str(filepath)
            else:
                print(
                    f"Warning: URL {image_url} returned non-image content: {content_type}, length: {content_length}")
        return response
