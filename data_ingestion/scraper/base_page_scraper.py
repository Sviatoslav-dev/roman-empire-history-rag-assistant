from bs4 import BeautifulSoup


class BasePageScraper:

    def __init__(self, html: str, title: str) -> None:
        """Initialize scraper with page HTML content."""
        self.soup = BeautifulSoup(html, "html.parser")
        self.title = title
        self.url = self.soup.find("link", rel="canonical")["href"]
