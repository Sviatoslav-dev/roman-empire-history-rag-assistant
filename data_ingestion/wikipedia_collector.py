from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import unquote, urlparse
from typing import List

from dotenv import load_dotenv

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.wikipedia_loader import WikipediaLoader
from data_ingestion.wikipedia_storage import WikipediaStorage

from data_ingestion.wikipedia_api_client import WikipediaApiClient
from logger import get_logger

load_dotenv()

logger = get_logger(__name__)

ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "./data/articles"))
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "./data/images"))


class WikipediaCollector:
    """High-level orchestration of Wikipedia article collection workflow."""

    MIN_ARTICLE_LENGTH = 2000  # Minimum number of characters in the articles
    MIN_ARTICLE_CITATIONS_NUMBER = 3 # Minimum number of citations in the articles

    def __init__(self, wikipedia_loader: WikipediaLoader, wikipedia_storage: WikipediaStorage) -> None:
        self.loader = wikipedia_loader
        self.storage = wikipedia_storage

        self.wikipedia_client = WikipediaApiClient()


    def filter_articles(self, articles: List[WikipediaArticleScraper]) -> List[WikipediaArticleScraper]:
        """Filter article scrapers using quality criteria.

        Args:
            articles: List of WikipediaArticleScraper instances to evaluate.

        Returns:
            A list of scrapers that passed the quality filters.
        """
        filtered: List[WikipediaArticleScraper] = []

        for article in articles:
            try:
                if article.passes_quality_filters(
                        self.MIN_ARTICLE_LENGTH,
                        self.MIN_ARTICLE_CITATIONS_NUMBER,
                ):
                    filtered.append(article)
            except Exception as e:
                logger.error("Filter error for %s: %s", article.title, e)
        return filtered

    def collect_articles(self, categories_file: str) -> List[WikipediaArticleScraper]:
        """Load category names from a file, download articles, and filter them.

        Args:
            categories_file: Path to newline-delimited category file.

        Returns:
            A list of WikipediaArticleScraper instances that passed filters.
        """
        categories = self.loader.load_categories(categories_file)

        article_titles = self.loader.get_all_articles_from_categories(categories)

        if categories:
            logger.info("Found %d articles in categories %s", len(article_titles), categories)
        else:
            logger.warning("No valid categories provided in %s.", categories_file)

        if not article_titles:
            logger.warning("No article titles discovered from categories; aborting fetch.")
            return []

        self.loader.fetch_pages_by_titles(article_titles)

        downloaded_articles = self.storage.get_downloaded_articles()
        filtered_articles = self.filter_articles(downloaded_articles)

        failed = [r for r in downloaded_articles if r not in filtered_articles]
        logger.info("Articles scanned: %d; passed: %d; failed: %d", len(downloaded_articles), len(filtered_articles), len(failed))
        return filtered_articles

        # chunks = client.split_articles_into_chunks(passed)
        # logger.info("Total article chunks created: %d", len(chunks))
        # client.download_images([image for chunk in chunks for image in chunk["images"]])


    def split_articles_into_chunks(self, articles: List[WikipediaArticleScraper]) -> List[Dict]:
        chunks = []
        for article in articles:
            sections = article.split_by_sections()
            for section in sections:
                chunk = {
                    "page_title": article.title,
                    "page_url": article.url,
                    "section_title": section.get("title"),
                    "section_path": section.get("title_path"),
                    "section_level": section.get("level"),
                    "text": section.get("text"),
                }
                chunks.append(chunk)
        return chunks

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

            # Extract filename from URL
            filename = image_url.split("/")[-1].split("?")[0]  # Remove query params
            # If filename is still a File: reference, extract the actual filename
            if "File:" in filename:
                filename = filename.split("File:")[-1]

            # Clean filename - remove invalid characters
            filename = filename.replace(" ", "_").replace(":", "_")
            # Remove any remaining query parameters or fragments
            filename = filename.split("?")[0].split("#")[0]

            # Ensure we have a valid extension
            if not any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"]):
                # Try to get extension from URL or default to jpg
                if "." in image_url:
                    ext = image_url.split(".")[-1].split("?")[0].split("/")[0]
                    if ext.lower() in ["jpg", "jpeg", "png", "gif", "webp", "svg"]:
                        filename = f"{filename}.{ext}"
                    else:
                        filename = f"{filename}.jpg"
                else:
                    filename = f"{filename}.jpg"

            filename = self._safe_filename(filename, max_length=255)

            filepath = IMAGES_DIR / filename

            # Skip if already exists
            if filepath.exists():
                print(f"File already exists: {filepath}")
                continue


            filename = self._extract_image_filename(image_url)
            metadata = self.wikipedia_client.get_image_license(filename)

            # print(f"{metadata=}")

            if metadata and self.should_skip_image(metadata):
                print("Skip image: fair use")
                continue

            # self.wikipedia_client.download_image(image_url, filepath)
            # time.sleep(0.5)

    def normalize(self, text: str) -> str:
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
        combined = self.normalize(combined_raw)

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

    def _safe_filename(self, title: str, max_length: int | None = None) -> str:
        """Return a filesystem-safe filename (without extension) for a title."""
        safe_name = unquote(title).replace(" ", "_")
        safe_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in safe_name)
        if max_length and len(safe_name) > max_length:
            name, ext = os.path.splitext(safe_name)
            h = hashlib.md5(safe_name.encode("utf-8")).hexdigest()[:8]
            cut_len = max_length - len(ext) - len(h) - 1  # 1 for "_"
            safe_name = f"{name[:cut_len]}_{h}{ext}"
        return safe_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wikipedia client helper: download and filter saved article HTML files.")
    parser.add_argument(
        "--categories-file",
        type=str,
        default=os.getenv("CATEGORIES_FILE", "categories.txt"),
        help="Path to a text file with Wikipedia categories (one per line)"
    )


    args = parser.parse_args()

    loader = WikipediaLoader()
    storage = WikipediaStorage()
    collector = WikipediaCollector(loader, storage)
    articles = collector.collect_articles(args.categories_file)

    if articles:
        logger.info("Articles that passed the filters:")
        for p in articles:
            logger.info("- %s", p.title)
