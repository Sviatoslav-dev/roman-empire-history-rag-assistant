import os
import re
import time
from pathlib import Path
from typing import Set, List
from urllib.parse import urlparse, unquote

from dotenv import load_dotenv
from tqdm import tqdm

from data_ingestion.scraper.wikipedia_category_scraper import WikipediaCategoryScraper
from data_ingestion.wikipedia_api_client import WikipediaApiClient
from data_ingestion.wikipedia_storage import WikipediaStorage
from logger import get_logger

load_dotenv()

logger = get_logger(__name__)

_wikipedia_client = WikipediaApiClient()
_storage = WikipediaStorage()

ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "./data/articles"))
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "./data/images"))

class WikipediaLoader:
    """Coordinates loading Wikipedia data from external sources."""

    RATE_LIMIT_DELAY = 0.5

    def get_all_articles_from_categories(self, categories: List[str]) -> Set[str]:
        """Collect article titles from the given Wikipedia categories.

        Args:
            categories: List of category names to crawl.

        Returns:
            A set of article titles (strings) discovered in the categories.
        """
        articles = set()

        for category in categories:
            logger.info(f"Crawling category: {category}")

            html = _wikipedia_client.fetch_category(category)
            if not html:
                logger.warning(f"Failed to fetch articles from category: {category}; skipping")
                continue

            scraper = WikipediaCategoryScraper(html, category)
            articles.update(scraper.extract_articles_from_category())

        return articles

    def fetch_pages_by_titles(self, titles: Set[str]) -> List[str]:
        """Download multiple Wikipedia articles as HTML files into `ARTICLES_DIR`.

        Args:
            titles: A set of article titles to fetch.

        Returns:
            A list of downloaded page HTML contents (strings).
        """

        pages: List[str] = []
        for title in tqdm(titles, desc="Downloading Wikipedia articles"):
            if _storage.article_exists(title):
                logger.info(f"Skipping {title} as it already exists.")
                continue

            page_html = _wikipedia_client.fetch_article(title)
            if page_html is None:
                logger.warning("Failed to fetch article: %s", title)
                time.sleep(self.RATE_LIMIT_DELAY)
                continue

            _storage.save_article_to_file(title, page_html)
            pages.append(page_html)

            time.sleep(self.RATE_LIMIT_DELAY)  # polite rate limiting
        return pages

    def load_categories(self, categories_file: str) -> List[str]:
        """Load categories names from a newline-delimited text file.

        Args:
            categories_file: Path to a file containing category names, one per line.

        Returns:
            A list of category names (strings).
        """
        with open(categories_file, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


    def download_images(self, urls: List[str]):
        """
        Download an image from Wikipedia / Wikimedia.

        Handles Wikipedia image URLs which may be:
        - Thumbnail URLs (need conversion to full-size)
        - File: URLs (need conversion to actual image URL)
        - Direct image URLs
        """
        for image_url in tqdm(urls, desc="Downloading images"):
            print("Processing image URL:", image_url)
            # Convert Wikipedia thumbnail URLs to full-size
            # Format: .../thumb/a/ab/Filename.jpg/220px-Filename.jpg -> .../a/ab/Filename.jpg
            if "/thumb/" in image_url:
                image_url = self._convert_thumbnail_to_fullsize(image_url)
            # Handle File: URLs - these are page URLs, not direct image URLs
            # We should skip these as they require API calls to get the actual image URL
            if "/wiki/File:" in image_url or "/wiki/Image:" in image_url or "/w/extensions/wikihiero" in image_url:
                # These are page URLs, not direct image URLs
                # Skip them for now (would need MediaWiki API to resolve)
                continue

            # Validate URL before processing
            if not image_url or not isinstance(image_url, str):
                continue
            # Check for malformed URLs (duplicate filenames in path)
            url_parts = image_url.split("/")
            if len(url_parts) >= 2:
                last_two = url_parts[-2:]
                # If last two parts are the same (except for query params), it's malformed
                if last_two[0].split("?")[0] == last_two[1].split("?")[0] and last_two[0].split("?")[0]:
                    # Remove the duplicate
                    image_url = "/".join(url_parts[:-1])

            # Normalize the URL
            if image_url.startswith("//"):
                image_url = "https:" + image_url
            elif image_url.startswith("/"):
                image_url = "https://en.wikipedia.org" + image_url

            # # Extract filename from URL
            # filename = image_url.split("/")[-1].split("?")[0]  # Remove query params
            # # If filename is still a File: reference, extract the actual filename
            # if "File:" in filename:
            #     filename = filename.split("File:")[-1]
            #
            # # Clean filename - remove invalid characters
            # filename = filename.replace(" ", "_").replace(":", "_")
            # # Remove any remaining query parameters or fragments
            # filename = filename.split("?")[0].split("#")[0]
            #
            # # Ensure we have a valid extension
            # if not any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"]):
            #     # Try to get extension from URL or default to jpg
            #     if "." in image_url:
            #         ext = image_url.split(".")[-1].split("?")[0].split("/")[0]
            #         if ext.lower() in ["jpg", "jpeg", "png", "gif", "webp", "svg"]:
            #             filename = f"{filename}.{ext}"
            #         else:
            #             filename = f"{filename}.jpg"
            #     else:
            #         filename = f"{filename}.jpg"

            image_title = self._extract_image_filename(image_url)

            # Skip if already exists
            if _storage.image_exists(image_title):
                print(f"image {image_title} already exists")
                continue


            # print(f"{metadata=}")
            metadata = _wikipedia_client.get_image_license(image_title)

            if metadata and self.should_skip_image(metadata):
                print("Skip image: fair use")
                continue

            _wikipedia_client.download_image(image_url, _storage.image_filepath(image_title))
            time.sleep(0.5)

    def normalize_license(self, text: str) -> str:
        """
        Normalize license text:
        - lowercase
        - replace separators with spaces
        - collapse multiple spaces
        """
        text = text.lower()
        text = re.sub(r"[-_/]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def should_skip_image(self, extmetadata: dict) -> bool:
        license_name = extmetadata.get("LicenseShortName", {}).get("value", "")
        usage_terms = extmetadata.get("UsageTerms", {}).get("value", "")

        combined_raw = f"{license_name} {usage_terms}"
        combined = self.normalize_license(combined_raw)

        print(f"{combined=}")

        forbidden_triggers = [
            # fair use / non-free
            "fair",
            "non free",
            "nonfree",

            # copyright
            "copyright",
            "all rights reserved",

            # Creative Commons restrictions
            "nc",  # non-commercial
            "noncommercial",
            "nd",  # no-derivatives
            "no derivatives",
        ]

        tokens = combined.split()

        for trigger in forbidden_triggers:
            if " " in trigger:
                if trigger in combined:
                    return True
            else:
                if trigger in tokens:
                    return True

        return False

    def _convert_thumbnail_to_fullsize(self, image_url: str) -> str:
        parts = image_url.split("/thumb/")
        if len(parts) == 2:
            base_url = parts[0]  # e.g., "https://upload.wikimedia.org/wikipedia/commons"
            thumb_path = parts[1]  # e.g., "a/ab/Filename.jpg/220px-Filename.jpg"

            # Split the thumb path into segments
            segments = thumb_path.split("/")

            # The structure is: hash1/hash2/filename/size-filename
            # We need: hash1/hash2/filename
            if len(segments) >= 3:
                # Take everything except the last segment (which is the size-prefixed filename)
                hash_and_file = "/".join(segments[:-1])
                image_url = f"{base_url}/{hash_and_file}"
            else:
                # Fallback: try to remove size prefix from last segment
                if len(segments) == 2:
                    filename_with_size = segments[-1]
                    # Check if it has a size prefix like "220px-"
                    if "-" in filename_with_size:
                        # Try to extract original filename
                        # The size prefix is usually at the start: "220px-Filename.jpg"
                        parts_filename = filename_with_size.split("-", 1)
                        if len(parts_filename) == 2 and "px" in parts_filename[0]:
                            # This is a size-prefixed filename, use the original from previous segment
                            image_url = f"{base_url}/{segments[0]}"
                        else:
                            # No clear size prefix, use as-is
                            image_url = f"{base_url}/{thumb_path}"
                    else:
                        image_url = f"{base_url}/{thumb_path}"
                else:
                    # Single segment, use as-is
                    image_url = f"{base_url}/{thumb_path}"
        return image_url

    def _extract_image_filename(self, image_url: str) -> str:
        path = urlparse(image_url).path
        return unquote(os.path.basename(path))
