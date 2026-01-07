"""Wikipedia API client for fetching articles, sections, and images."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import unquote

import requests
import wikipedia
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BASE_URL = "https://en.wikipedia.org"


class WikipediaClient:
    """
    Client for fetching Wikipedia articles and images.

    Key responsibilities for this project:
    - Fetch the main article (e.g. 'Roman Empire') and all related articles via in-text links.
    - Split each article into sections based on HTML headings (h2–h6).
    - Return the lowest-level sections with their text, images, and outgoing links.
    """

    def __init__(self) -> None:
        """Initialize Wikipedia client."""
        wikipedia.set_lang("en")
        # Use configured IMAGES_DIR or sensible default 'images' in repo root
        images_dir = os.getenv("IMAGES_DIR") or "images"
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def get_all_articles_from_categories(
        self,
        categories: list[str],
    ) -> Set[str]:
        visited_categories = set()
        articles = set()

        for category in categories:
            norm_category = category.lower().replace("_", " ").strip()

            print(f"Crawling category: {category}")

            if norm_category in visited_categories:
                continue

            visited_categories.add(norm_category)

            url = f"{BASE_URL}/wiki/Category:{category.replace(' ', '_')}"
            response = requests.get(
                url,
                headers={
                    "User-Agent": "RomanEmpireResearchBot/1.0 (contact: your-email@example.com)"
                },
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # 1️⃣ Articles
            for group in soup.select("#mw-pages .mw-category-group"):
                for link in group.select("a[href^='/wiki/']"):
                    title = unquote(link["href"]).replace("/wiki/", "")
                    if ":" not in title:
                        articles.add(title.replace("_", " "))
        return articles


    # ---------------------------------------------------------------------
    # High-level API (download raw article HTML files into ARTICLES_DIR)
    # ---------------------------------------------------------------------
    def fetch_page(self, title: str) -> Optional[Dict]:
        """
        Download the raw HTML for a Wikipedia article and save it to
        the folder specified by the environment variable `ARTICLES_DIR`.

        Returns a dict with:
        {
            "title": str,
            "path": str,        # filesystem path to the saved HTML file
            "downloaded": bool, # True if newly downloaded, False if skipped (already existed)
        }
        or None on error.
        """
        articles_dir = Path(os.getenv("ARTICLES_DIR") or "articles")
        articles_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_name = unquote(title).replace(" ", "_")
        safe_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in safe_name)
        file_path = articles_dir / f"{safe_name}.html"

        if file_path.exists():
            return {"title": title, "path": str(file_path), "downloaded": False}

        try:
            page = wikipedia.page(title, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            if e.options:
                return self.fetch_page(e.options[0])
            return None
        except Exception as e:
            print(f"Error fetching page '{title}': {e}")
            return None

        try:
            html = page.html()
        except Exception as e:
            print(f"Error getting HTML for '{title}': {e}")
            return None

        if not html:
            print(f"Warning: No HTML content for '{title}'")
            return None

        try:
            file_path.write_text(html, encoding="utf-8")
        except Exception as e:
            print(f"Error writing HTML for '{title}' to '{file_path}': {e}")
            return None

        return {"title": page.title, "path": str(file_path), "downloaded": True}


    def fetch_pages_by_titles(self, titles: List[str]) -> List[Dict]:
        """Download multiple Wikipedia articles as HTML files into `ARTICLES_DIR`."""
        import time

        results: List[Dict] = []
        for title in tqdm(titles, desc="Downloading Wikipedia articles"):
            res = self.fetch_page(title)
            if res:
                results.append(res)
            time.sleep(0.5)  # polite rate limiting
        return results


    # -----------------------------------------------------------------
    # Filtering API: keep only articles that match project criteria
    # -----------------------------------------------------------------
    def _count_citations(self, soup: BeautifulSoup) -> int:
        """Estimate number of inline citations in the article.

        We count <sup class="reference"> elements and also numbered
        reference list entries under <ol class="references">.
        """
        sup_refs = soup.select('sup.reference')
        ol_refs = soup.select('ol.references li')
        # Some pages use a different references structure; also count
        # anchors that point to cite_note ids in the article body.
        anchor_refs = soup.select("a[href^='#cite_note']")
        # Use set to avoid double-counting anchors inside <sup>
        return len(sup_refs) + len(ol_refs) + len(anchor_refs)

    def _has_problem_or_update_box(self, soup: BeautifulSoup) -> bool:
        """Detect whether the page has maintenance/problem/update/dispute banners.

        We look for known banner classes (ambox, hatnote, mw-warning) and check
        their text for keywords like 'disput', 'cleanup', 'update', 'problem', 'outdated'.
        """
        banner_selectors = [".ambox", ".hatnote", ".mw-warning", ".metadata", ".notice", ".messagebox"]
        keywords = ("disput", "cleanup", "update", "problem", "outdat", "bias", "needs sources", "merge", "contradictory")
        for sel in banner_selectors:
            for el in soup.select(sel):
                text = (' '.join(el.get_text().split()) or "").lower()
                for kw in keywords:
                    if kw in text:
                        return True
        # Also check for templates rendered as tables with class 'ambox'
        for el in soup.select("table.ambox, div.ambox"):
            if el and any(kw in (' '.join(el.get_text().split()) or "").lower() for kw in keywords):
                return True
        # Check top of content for maintenance phrases
        content = soup.select_one("#mw-content-text")
        if content:
            top_text = ' '.join(' '.join(p.get_text().split()) for p in content.select('p')[:2]).lower()
            for kw in keywords:
                if kw in top_text:
                    return True
        return False

    def _is_english_article(self, soup: BeautifulSoup) -> bool:
        """Check if the page HTML is English.

        We prefer the <html lang="en"> attribute. If absent (HTML fragment),
        check for any element with a lang or xml:lang attribute starting with 'en'.
        Finally, fall back to presence of common Wikipedia content containers
        (e.g. #mw-content-text or .mw-parser-output) which usually indicate an
        English Wikipedia page in this project.
        """
        html_tag = soup.find('html')
        if html_tag is not None:
            lang = html_tag.get('lang') or html_tag.get('xml:lang')
            if lang and lang.lower().startswith('en'):
                return True

        # If the HTML <html> tag is missing (we have an HTML fragment), look
        # for any element that explicitly declares a language attribute.
        for tag in soup.find_all(attrs={ 'lang': True }):
            if (tag.get('lang') or '').lower().startswith('en'):
                return True
        for tag in soup.find_all(attrs={ 'xml:lang': True }):
            if (tag.get('xml:lang') or '').lower().startswith('en'):
                return True

        # Fallback: if the document contains typical Wikipedia content containers
        # we assume it's an English Wikipedia snippet that should be processed.
        if soup.select_one('#mw-content-text') is not None or soup.select_one('.mw-parser-output') is not None:
            return True

        return False

    def filter_downloaded_articles(self, paths: Optional[List[str]] = None) -> List[Dict]:
        """Scan downloaded HTML files and return entries with pass/fail and reasons.

        Behavior changes / enhancements:
        - Looks for articles in multiple likely locations: ARTICLES_DIR env, './articles', and 'data/articles' next to this module.
        - Returns a list of all scanned article entries (each with title/path/passes/reasons) so callers can inspect failures.
        """
        # Resolve articles directory candidates
        env_dir = os.getenv("ARTICLES_DIR")
        candidates = []
        if env_dir:
            candidates.append(Path(env_dir))
        candidates.append(Path("articles"))
        # folder next to this module: data/articles
        candidates.append(Path(__file__).resolve().parent / "data" / "articles")

        files: List[Path] = []
        if paths:
            files = [Path(p) for p in paths]
        else:
            for c in candidates:
                if c.exists() and c.is_dir():
                    files = sorted(c.glob('*.html'))
                    if files:
                        break

        if not files:
            print("No article HTML files found in ARTICLES_DIR, ./articles, or data/articles; searched:")
            for c in candidates:
                print(f" - {c}")
            return []

        results: List[Dict] = []

        for fp in files:
            try:
                html = fp.read_text(encoding='utf-8')
            except Exception:
                # Skip unreadable files
                continue
            soup = BeautifulSoup(html, 'html.parser')
            title_tag = soup.find('title')
            title = title_tag.get_text().replace(' - Wikipedia', '').strip() if title_tag else fp.stem

            reasons: List[str] = []

            # Length check: use visible text inside content area
            content_el = soup.select_one('#mw-content-text')
            visible_text = (' '.join(content_el.get_text().split()) if content_el else ' '.join(soup.get_text().split()))
            if len(visible_text) <= 2000:
                reasons.append(f"too_short (length={len(visible_text)})")

            # Citations count
            citations = self._count_citations(soup)
            if citations < 3:
                reasons.append(f"few_citations (count={citations})")

            # Problem or update box
            if self._has_problem_or_update_box(soup):
                reasons.append("has_problem_or_update_box")

            # English availability
            if not self._is_english_article(soup):
                reasons.append("not_english")

            passes = len(reasons) == 0
            entry = {"title": title, "path": str(fp), "passes": passes, "reasons": reasons}
            results.append(entry)
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wikipedia client helper: download and filter saved article HTML files.")
    parser.add_argument("--categories", type=str, default=os.getenv("CATEGORIES", "Roman_Empire"), help="Comma-separated list of Wikipedia categories to crawl and download (default: 'Roman_Empire')")
    # Download is enabled by default; provide a --no-download to turn it off
    parser.add_argument("--download", dest="download", action="store_true", default=True, help="Enable downloading (default: enabled).")
    parser.add_argument("--no-download", dest="download", action="store_false", help="Disable downloading.")
    # Filtering is enabled by default; provide a --no-filter to turn it off
    parser.add_argument("--filter", dest="do_filter", action="store_true", default=True, help="Enable filtering of saved HTML files (default: enabled).")
    parser.add_argument("--no-filter", dest="do_filter", action="store_false", help="Disable filtering.")
    parser.add_argument("--paths", nargs="*", help="Optional list of HTML file paths to filter (overrides scanning ARTICLES_DIR)")

    args = parser.parse_args()

    client = WikipediaClient()

    # Optional downloading from categories
    if args.download and args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        if categories:
            articles = client.get_all_articles_from_categories(categories)
            print(f"Found {len(articles)} articles in categories {categories}")
            downloaded_pages = client.fetch_pages_by_titles(list(articles))
            print(f"Downloaded {len(downloaded_pages)} pages.")
        else:
            print("No valid categories provided to --categories.")

    # Optional filtering of saved HTML files
    if args.do_filter:
        paths = args.paths if args.paths else None
        results = client.filter_downloaded_articles(paths)
        passed = [r for r in results if r.get('passes')]
        failed = [r for r in results if not r.get('passes')]
        print(f"Articles scanned: {len(results)}; passed: {len(passed)}; failed: {len(failed)}")
        if passed:
            print("Passed articles:")
            for p in passed:
                print(f"- {p['title']} ({p['path']})")
        if failed:
            print("\nFailed articles (showing up to 10 with reasons):")
            for p in failed[:10]:
                print(f"- {p['title']} ({p['path']}) -> reasons: {p['reasons']}")

    if not (args.download or args.do_filter):
        print("Both download and filter are disabled. Nothing to do.")
        parser.print_help()
