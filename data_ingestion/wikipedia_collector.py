from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from tqdm import tqdm

from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.scraper.wikipedia_category_scraper import WikipediaCategoryScraper
from data_ingestion.wikipedia_api_client import WikipediaApiClient
from logger import get_logger

load_dotenv()

logger = get_logger(__name__)

ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "./data/articles"))
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "./data/images"))



class WikipediaCollector:
    """
    High-level orchestration layer for collecting Wikipedia articles.

    Responsibilities:
    - collect article titles from Wikipedia categories,
    - download raw HTML pages for articles,
    - load saved article HTML files,
    - apply quality filters to downloaded articles.
    """

    def __init__(self) -> None:
        """Initialize Wikipedia client."""
        self.wikipedia_client = WikipediaApiClient()

    def get_all_articles_from_categories(self, categories: list[str]) -> Set[str]:
        """Collect article titles from the given Wikipedia categories."""
        articles = set()

        for category in categories:
            logger.info(f"Crawling category: {category}")

            # extract articles via helper
            scraper = WikipediaCategoryScraper.get_by_title(category)
            articles.update(scraper.extract_articles_from_category())

        return articles


    def fetch_page(self, title: str) -> Optional[Dict]:
        """
        Download the raw HTML for a Wikipedia article and save it to
        the folder specified by the environment variable `ARTICLES_DIR`.
        """
        ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

        safe_name = self._safe_filename(title)
        file_path = ARTICLES_DIR / f"{safe_name}.html"

        if file_path.exists():
            return {"title": title, "path": str(file_path), "downloaded": False}

        page_html = self.wikipedia_client.fetch_article(title)

        file_path.write_text(page_html, encoding="utf-8")

        return {"title": title, "path": str(file_path), "downloaded": True}


    def fetch_pages_by_titles(self, titles: List[str]) -> List[Dict]:
        """Download multiple Wikipedia articles as HTML files into `ARTICLES_DIR`."""

        results: List[Dict] = []
        for title in tqdm(titles, desc="Downloading Wikipedia articles"):
            res = self.fetch_page(title)
            if res:
                results.append(res)
            time.sleep(0.5)  # polite rate limiting
        return results

    def get_downloaded_articles(self) -> List[WikipediaArticleScraper]:
        """
        Load saved Wikipedia article HTML files from `ARTICLES_DIR`
        and return scraper instances for them.
        """

        articles: List[WikipediaArticleScraper] = []
        files: List[Path] = [ARTICLES_DIR]

        for c in files:
            if not c.exists() or not c.is_dir():
                continue
            for fp in sorted(c.glob("*.html")):
                articles.append(WikipediaArticleScraper.get_from_file(fp))
        return articles

    def filter_articles(self, articles: List[WikipediaArticleScraper]) -> List[WikipediaArticleScraper]:
        """Filter article scrapers using quality criteria."""
        return [article for article in articles if article.passes_quality_filters()]

    def split_articles_into_chunks(self, articles: List[WikipediaArticleScraper]) -> List[Dict]:
        chunks = []
        for article in articles:
            chunks.extend(article.split_by_sections())
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

            filepath = IMAGES_DIR / filename

            # Skip if already exists
            if filepath.exists():
                print(f"File already exists: {filepath}")
                continue


            filename = self._extract_image_filename(image_url)
            metadata = self.wikipedia_client.get_image_license(filename)

            print(f"{metadata=}")

            if metadata and self.should_skip_image(metadata):
                print("Skip image: fair use")
                continue

            self.wikipedia_client.download_image(image_url, filepath)
            time.sleep(0.5)

    def should_skip_image(self, extmetadata: dict) -> bool:
        license_name = extmetadata.get("LicenseShortName", {}).get("value", "").lower()
        usage_terms = extmetadata.get("UsageTerms", {}).get("value", "").lower()

        forbidden_keywords = [
            "fair use",
            "non-free",
            "copyright",
            "cc-by-nc",
            "cc by-nc",
            "noncommercial",
            "cc-by-nd",
            "no derivatives",
            "nd"
        ]

        combined = f"{license_name} {usage_terms}"

        return any(keyword in combined for keyword in forbidden_keywords)

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

    def _safe_filename(self, title: str) -> str:
        """Return a filesystem-safe filename (without extension) for a title."""
        safe_name = unquote(title).replace(" ", "_")
        safe_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in safe_name)
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
    # Download is enabled by default; provide a --no-download to turn it off
    parser.add_argument("--download", dest="download", action="store_true", default=True, help="Enable downloading (default: enabled).")
    parser.add_argument("--no-download", dest="download", action="store_false", help="Disable downloading.")
    # Filtering is enabled by default; provide a --no-filter to turn it off
    parser.add_argument("--filter", dest="do_filter", action="store_true", default=True, help="Enable filtering of saved HTML files (default: enabled).")
    parser.add_argument("--no-filter", dest="do_filter", action="store_false", help="Disable filtering.")

    args = parser.parse_args()

    client = WikipediaCollector()

    # Optional downloading from categories
    if args.download and args.categories_file:
        with open(args.categories_file, encoding="utf-8") as f:
            categories = [line.strip() for line in f if line.strip()]

        if categories:
            articles = client.get_all_articles_from_categories(categories)
            logger.info("Found %d articles in categories %s", len(articles), categories)
            downloaded_pages = client.fetch_pages_by_titles(list(articles))
            logger.info("Downloaded %d pages.", len(downloaded_pages))
        else:
            logger.warning("No valid categories provided to --categories.")

    # Optional filtering of saved HTML files
    if args.do_filter:
        downloaded_articles = client.get_downloaded_articles()
        passed = client.filter_articles(downloaded_articles)
        failed = [r for r in downloaded_articles if r not in passed]
        logger.info("Articles scanned: %d; passed: %d; failed: %d", len(downloaded_articles), len(passed), len(failed))
        if passed:
            logger.info("Passed articles:")
            for p in passed:
                logger.info("- %s", p.title)

        if failed:
            logger.info("\nFailed articles (showing up to 10 with reasons):")
            for p in failed[:10]:
                logger.info("- %s", p.title)

    if not (args.download or args.do_filter):
        logger.warning("Both download and filter are disabled. Nothing to do.")
        parser.print_help()

    chunks = client.split_articles_into_chunks(client.get_downloaded_articles())
    logger.info("Total article chunks created: %d", len(chunks))
    client.download_images([image for chunk in chunks for image in chunk["images"]])
