from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from urllib.parse import unquote, urlparse

from data_ingestion.wikipedia_api_client import WikipediaApiClient

_wikipedia_client = WikipediaApiClient()


@dataclass(slots=True)
class WikipediaImage:
    """Work with a single Wikipedia/Wikimedia image URL + related metadata."""

    url: str
    local_path: Path | None = None

    def convert_thumbnail_to_fullsize(self) -> None:
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

    def normalize_url(self, *, wikipedia_host: str = "https://en.wikipedia.org") -> "WikipediaImage":
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
        return self

    def get_filename(self) -> str:
        """Extract the final path segment (decoded) from the image URL."""
        path = urlparse(self.url).path
        return unquote(os.path.basename(path))
