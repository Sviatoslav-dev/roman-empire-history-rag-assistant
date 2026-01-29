import os
from typing import Optional

import time
from pathlib import Path
import tempfile

import requests

import wikipedia
from dotenv import load_dotenv
from wikipedia import PageError, HTTPTimeoutError, WikipediaException

from logger import get_logger

logger = get_logger(__name__)

load_dotenv()
USER_AGENT_EMAIL = os.getenv("USER_AGENT_EMAIL")

class WikipediaApiClient:
    """Client for fetching raw Wikipedia HTML pages."""

    BASE_URL = "https://en.wikipedia.org"
    BASE_API_URL = "https://en.wikipedia.org/w/api.php"
    HEADERS = {
        "User-Agent": f"RomanEmpireResearchBot/1.0 (contact: {USER_AGENT_EMAIL})",
    }
    REQUEST_TIMEOUT = 10  # seconds

    def __init__(self) -> None:
        """Initialize the Wikipedia API client and set default request headers."""
        wikipedia.set_lang("en")

    def fetch_article(self, title) -> Optional[str]:
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
        except (PageError, HTTPTimeoutError, WikipediaException, KeyError) as e:
            logger.error("Error fetching page '%s': %s", title, e)
            return None

        response = requests.get(page.url, headers=self.HEADERS)
        response.raise_for_status()
        html = response.text

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

    def get_image_license(self, filename: str) -> dict | None:
        params = {
            "action": "query",
            "format": "json",
            "titles": f"File:{filename}",
            "prop": "imageinfo",
            "iiprop": "extmetadata"
        }

        r = requests.get(self.BASE_API_URL, params=params, headers=self.HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), None)

        if not page or "imageinfo" not in page:
            return None

        return page["imageinfo"][0]["extmetadata"]

    def download_image(self, image_url: str, filepath: Path) -> str | requests.Response:
        response = requests.get(image_url, headers=self.HEADERS, timeout=30, stream=True)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            time.sleep(int(retry_after))
            response = requests.get(image_url, headers=self.HEADERS, timeout=30, stream=True)
            response.raise_for_status()

        if response.status_code == 200:
            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            content_length = response.headers.get("content-length", "0")
            # Accept if it's an image or has content
            if "image" in content_type or (content_length and int(content_length) > 0):
                filepath.parent.mkdir(parents=True, exist_ok=True)

                fd, tmp_name = tempfile.mkstemp(prefix=filepath.name, dir=str(filepath.parent))
                try:
                    with os.fdopen(fd, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_name, filepath)
                    return str(filepath)
                except Exception:
                    try:
                        os.unlink(tmp_name)
                    except Exception:
                        pass
                    raise
            else:
                logger.warning(
                    "URL returned non-image content: url=%s content_type=%s content_length=%s",
                    image_url,
                    content_type,
                    content_length,
                )
        return response
