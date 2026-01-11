from __future__ import annotations

from typing import Any
from bs4 import BeautifulSoup


class BasePageScraper:
    """
    Base class for Wikipedia page scrapers.

    Provides common HTML parsing setup and shared attributes
    for concrete scraper implementations.
    """

    def __init__(self, html: str, title: str) -> None:
        """Initialize scraper with page HTML content.

        Args:
            html: Raw HTML content to parse.
            title: The article or page title associated with the HTML.
        """
        self.soup: Any = BeautifulSoup(html, "html.parser")
        self.title: str = title
