from __future__ import annotations

import os
import re
from pathlib import Path

from urllib.parse import unquote, urlparse

from data_ingestion.wikipedia_api_client import WikipediaApiClient

_wikipedia_client = WikipediaApiClient()


class WikipediaImage:
    """Work with a single Wikipedia/Wikimedia image URL + related metadata.

    This class intentionally has no I/O.

    Usage:
        img = WikipediaImage(url).normalize_url()
        filename = img.get_filename()
        if img.is_license_allowed(extmetadata): ...
    """


    def __init__(self, url: str):
        self.url = url
        self.local_path: Path | None = None

    def convert_thumbnail_to_fullsize(self):
        """Convert a Wikimedia thumbnail URL to the original full-size file URL."""

        parts = self.url.split("/thumb/")
        if len(parts) != 2:
            return

        base_url, thumb_path = parts
        segments = thumb_path.split("/")

        # Expected structure:
        #   hash1/hash2/filename/220px-filename
        # We need:
        #   hash1/hash2/filename
        if len(segments) >= 3:
            self.url = f"{base_url}/{'/'.join(segments[:-1])}"

    def normalize_url(self, *, wikipedia_host: str = "https://en.wikipedia.org"):
        """Normalize various Wikipedia image src forms into a direct absolute URL.

        Handles:
        - thumbnail -> full-size conversion
        - protocol-relative URLs ("//upload.wikimedia...")
        - root-relative URLs ("/wiki/..." or "/w/..." -> wikipedia_host prefix)
        - malformed URLs where the last path segment is duplicated
        """
        # Convert thumb -> fullsize for better dedup + retrieval
        if "/thumb/" in self.url:
            self.convert_thumbnail_to_fullsize()

        # Normalize malformed URLs (duplicate last segment)
        image_url = self.url
        url_parts = image_url.split("/")
        if len(url_parts) >= 2:
            last_two = url_parts[-2:]
            a = last_two[0].split("?")[0]
            b = last_two[1].split("?")[0]
            if a and a == b:
                image_url = "/".join(url_parts[:-1])

        # Normalize protocol/host
        if image_url.startswith("//"):
            image_url = "https:" + image_url
        elif image_url.startswith("/"):
            image_url = wikipedia_host.rstrip("/") + image_url

        self.url = image_url

    def get_filename(self) -> str:
        """Extract the final path segment (decoded) from the image URL."""
        path = urlparse(self.url).path
        return unquote(os.path.basename(path))

    @staticmethod
    def _normalize_license_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[-_/]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def is_license_allowed(self) -> bool:
        """Return True if the image license looks safe-to-use.

        We conservatively reject common non-free / fair-use indicators.
        When metadata is missing, we default to allowing.
        """
        extmetadata = _wikipedia_client.get_image_license(self.get_filename())

        license_name = (extmetadata.get("LicenseShortName") or {}).get("value", "")
        usage_terms = (extmetadata.get("UsageTerms") or {}).get("value", "")

        combined = self._normalize_license_text(f"{license_name} {usage_terms}")
        tokens = set(combined.split())

        forbidden_triggers: tuple[str, ...] = (
            "fair use",
            "fair",
            "non free",
            "nonfree",
            "copyright",
            "all rights reserved",
            "noncommercial",
            "no derivatives",
            "nc",
            "nd",
        )

        for trigger in forbidden_triggers:
            if " " in trigger:
                if trigger in combined:
                    return False
            else:
                if trigger in tokens:
                    return False

        return True

