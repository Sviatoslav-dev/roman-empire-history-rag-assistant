import json
from pathlib import Path
from typing import Optional, List, Dict, Set
from urllib.parse import unquote

from bs4 import BeautifulSoup

from data_ingestion.scraper.base_page_scraper import BasePageScraper
from data_ingestion.wikipedia_api_client import WikipediaApiClient
from logger import get_logger

logger = get_logger(__name__)


class WikipediaArticleScraper(BasePageScraper):

    @classmethod
    def get_by_title(cls, title: str) -> Optional['WikipediaArticleScraper']:
        """Fetch article by title and return scraper instance."""
        client = WikipediaApiClient()
        html = client.fetch_article(title)
        if html:
            return cls(html, title)
        return None

    @classmethod
    def get_from_file(cls, article_path: Path) -> Optional["WikipediaArticleScraper"]:
        """Create a WikipediaArticleScraper from a saved HTML file.

        Expects a direct path to an HTML file.
        The article title is inferred from the filename (without extension).
        """
        if not article_path.exists() or not article_path.is_file():
            logger.error("HTML file does not exist: %s", article_path)
            return None

        try:
            html = article_path.read_text(encoding="utf-8")
            title = article_path.stem  # filename without extension
            return cls(html, title)
        except Exception as e:
            logger.error("Error reading HTML file %s: %s", article_path, e)
            return None

    def passes_quality_filters(self) -> bool:
        """
        Check whether the article meets basic quality requirements.

        The article is considered valid if it:
        - contains sufficient visible text (more than 2000 characters),
        - has an adequate number of citations(3 and more),
        - does not include maintenance/problem/update banners,
        - represents an English Wikipedia article.

        Returns:
            bool: True if the article passes all quality filters,
                  False otherwise.
        """
        min_article_length = 2000  # characters
        min_citations_number = 3

        # Length check: use visible text inside content area
        content_el = self.soup.select_one("#mw-content-text")
        visible_text = (" ".join(content_el.get_text().split()) if content_el else " ".join(self.soup.get_text().split()))
        if len(visible_text) <= min_article_length:
            logger.info("Article %s failed filter: %s", self.title, f"too_short (length={len(visible_text)})")
            return False

        # Citations count
        citations = self.count_citations()
        if citations < min_citations_number:
            logger.info("Article %s failed filter: %s", self.title, f"few_citations (count={citations})")
            return False

        # Problem or update box
        if self.has_problem_or_update_box():
            logger.info("Article %s failed filter: %s", self.title, "has_problem_or_update_box")
            return False

        # English availability
        if not self.is_english_article():
            logger.info("Article %s failed filter: %s", self.title, "not_english")
            return False

        return True


    def count_citations(self) -> int:
        """
        Estimate the number of unique inline citations in the article.

        Counts unique citation anchors that reference Wikipedia `cite_note`
        entries (links with href starting with '#cite_note').
        """
        refs = set()
        for a in self.soup.select("a[href^='#cite_note']"):
            refs.add(a["href"])
        return len(refs)

    def has_problem_or_update_box(self) -> bool:
        """Detect whether the page has maintenance/problem/update/dispute banners.

        We look for known banner classes (ambox, hatnote, mw-warning) and check
        their text for keywords like 'disput', 'cleanup', 'update', 'problem', 'outdated'.
        """
        banner_selectors = [".ambox", ".hatnote", ".mw-warning", ".metadata", ".notice", ".messagebox"]
        keywords = ("disput", "cleanup", "update", "problem", "outdat", "bias", "needs sources", "merge", "contradictory")
        for sel in banner_selectors:
            for el in self.soup.select(sel):
                text = (' '.join(el.get_text().split()) or "").lower()
                for kw in keywords:
                    if kw in text:
                        return True
        # Also check for templates rendered as tables with class 'ambox'
        for el in self.soup.select("table.ambox, div.ambox"):
            if el and any(kw in (' '.join(el.get_text().split()) or "").lower() for kw in keywords):
                return True
        # Check top of content for maintenance phrases
        content = self.soup.select_one("#mw-content-text")
        if content:
            top_text = ' '.join(' '.join(p.get_text().split()) for p in content.select('p')[:2]).lower()
            for kw in keywords:
                if kw in top_text:
                    return True
        return False

    def is_english_article(self) -> bool:
        """Check if the page HTML is English.

        We prefer the <html lang="en"> attribute. If absent (HTML fragment),
        check for any element with a lang or xml:lang attribute starting with 'en'.
        Finally, fall back to presence of common Wikipedia content containers
        (e.g. #mw-content-text or .mw-parser-output) which usually indicate an
        English Wikipedia page in this project.
        """
        html_tag = self.soup.find('html')
        if html_tag is not None:
            lang = html_tag.get('lang') or html_tag.get('xml:lang')
            if lang and lang.lower().startswith('en'):
                return True

        # If the HTML <html> tag is missing (we have an HTML fragment), look
        # for any element that explicitly declares a language attribute.
        for tag in self.soup.find_all(attrs={ 'lang': True }):
            if (tag.get('lang') or '').lower().startswith('en'):
                return True
        for tag in self.soup.find_all(attrs={ 'xml:lang': True }):
            if (tag.get('xml:lang') or '').lower().startswith('en'):
                return True

        # Fallback: if the document contains typical Wikipedia content containers
        # we assume it's an English Wikipedia snippet that should be processed.
        if self.soup.select_one('#mw-content-text') is not None or self.soup.select_one('.mw-parser-output') is not None:
            return True

        return False

    def split_by_sections(self) -> List[Dict]:
        """
        Parse article HTML into a flat list of sections (split on h2–h6) and
        collect all outgoing links.

        Each heading defines a section; content until the next heading of the
        same or higher level is part of that section. This effectively splits
        the article by the lowest section level in the HTML.
        """
        # Remove table of contents
        toc = self.soup.find("div", id="toc")
        if toc:
            toc.decompose()

        for tag in self.soup.find_all(attrs={"role": ["navigation", "presentation"]}):
            tag.decompose()

        for tag in self.soup.find_all(class_="side-box"):
            tag.decompose()

        content_root = self.soup.find("div", id="mw-content-text")
        if content_root is None:
            # Try alternative selectors
            content_root = self.soup.find("div", class_="mw-parser-output")
            if content_root is None:
                print("Warning: Could not find content root in HTML")
                return [], set()

        sections: List[Dict] = []

        heading_stack: List[tuple[int, str]] = []
        current_section: Optional[Dict] = {
            "title": "Introduction",
            "title_path": "Introduction",
            "level": 1,
            "text_parts": [],
            "images": [],
        }

        # Iterate over all elements in the content area
        # Use direct children first, then descendants for nested content
        for el in content_root.find_all(["h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "div", "figure"]):
            if not getattr(el, "name", None):
                continue

            # New section starts at h2–h6
            if el.name in {"h2", "h3", "h4", "h5", "h6"}:
                level = int(el.name[1])
                title_text = el.get_text(" ", strip=True)
                if not title_text:
                    continue

                # Update heading stack to maintain hierarchy
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title_text))
                title_path = " > ".join(t for _, t in heading_stack)

                # Save previous section if it exists
                if current_section is not None:
                    # Finalize previous section
                    text_parts = current_section.pop("text_parts", [])
                    current_section["text"] = "\n\n".join(text_parts).strip()
                    if current_section["text"]:  # Only add non-empty sections
                        sections.append(current_section)

                current_section = {
                    "title": title_text,
                    "title_path": title_path,
                    "level": level,
                    "text_parts": [],
                    "images": [],
                }
                continue

            if el.name == "figure":
                # Finalize previous section
                text_parts = current_section.pop("text_parts", [])
                current_section["text"] = "\n\n".join(text_parts).strip()
                if current_section["text"]:  # Only add non-empty sections
                    sections.append(current_section)

                current_section = {
                    "title": current_section["title"],
                    "title_path": current_section["title_path"],
                    "level": current_section["level"],
                    "text_parts": [],
                    "images": [],
                }


            # Accumulate content into the current section
            if current_section is None:
                # Skip content before the first heading
                continue

            # Special handling for infobox tables
            # if el.name == "table":
            #     classes = el.get("class", [])
            #     class_str = " ".join(classes) if classes else ""
            #     if "infobox" in class_str.lower():
            #         # Extract structured infobox data
            #         infobox_text = self._extract_infobox_data(el, current_section)
            #         if infobox_text:
            #             current_section["text_parts"].append(infobox_text)
            #         continue

            if el.name == "table":
                classes = el.get("class", [])
                class_str = " ".join(classes).lower() if classes else ""

                table_json = self._extract_table_generic_json(el, current_section)

                if table_json:
                    current_section["text_parts"].append(
                        json.dumps(table_json, ensure_ascii=False)
                    )
                continue

            # Text content
            if el.name in {"p", "ul", "ol", "table"}:
                text = el.get_text(" ", strip=True)
                if text:
                    current_section["text_parts"].append(text)

            # Images in this element
            for img in el.select(":scope > a > img, :scope > span > a > img"):
                # Try multiple attributes for image source

                src = img.get("src") or img.get("data-src") or img.get("data-file-width") or img.get("href")
                if not src:
                    continue

                # Skip data URIs and very small images (likely icons)
                if src.startswith("data:") or "icon" in src.lower():
                    continue

                # Normalize protocol-relative URLs
                if src.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    src = "https://en.wikipedia.org" + src
                elif not src.startswith("http"):
                    src = "https://en.wikipedia.org" + src

                current_section["images"].append(src)

            # Links in this element
            for a in el.find_all("a", recursive=True):
                href = a.get("href") or ""
                # Only keep standard article links: /wiki/Title
                if not href.startswith("/wiki/"):
                    continue
                # Skip non-article namespaces like File:, Category:, etc.
                link_part = href.split("/wiki/")[1]
                if ":" in link_part:
                    continue
                title = unquote(link_part).replace("_", " ")

        # Finalize last section
        if current_section is not None:
            text_parts = current_section.pop("text_parts", [])
            current_section["text"] = "\n\n".join(text_parts).strip()
            if current_section["text"]:
                sections.append(current_section)

        # Filter out empty sections
        sections = [s for s in sections if s.get("text")]

        return sections

    def _extract_table_generic_json(self, table, current_section):
        def clean_text(el):
            return " ".join(el.stripped_strings)

        rows = table.find_all("tr")
        if not rows:
            return None

        caption = None
        header_rows = []
        data_rows = []

        # --- 1. Витягуємо caption (рядок з colspan на всю таблицю)
        first_row_cells = rows[0].find_all(["td", "th"])
        if len(first_row_cells) == 1 and first_row_cells[0].has_attr("colspan"):
            caption = clean_text(first_row_cells[0])
            rows = rows[1:]

        # --- 2. Збираємо header rows (поки є <th>)
        while rows and rows[0].find_all("th"):
            header_rows.append(rows.pop(0))

        # --- 3. Побудова багаторівневих заголовків
        header_matrix = []
        max_cols = 0

        for hr in header_rows:
            row = []
            for cell in hr.find_all("th"):
                text = clean_text(cell)
                colspan = int(cell.get("colspan", 1))
                row.extend([text] * colspan)
            max_cols = max(max_cols, len(row))
            header_matrix.append(row)

        # вирівнюємо всі рядки заголовків
        for row in header_matrix:
            if len(row) < max_cols:
                row.extend([""] * (max_cols - len(row)))

        # транспонуємо → отримуємо колонкові ієрархії
        columns = []
        for col_idx in range(max_cols):
            hierarchy = []
            for row in header_matrix:
                if row[col_idx]:
                    hierarchy.append(row[col_idx])
            columns.append(hierarchy)

        # --- 4. Парсинг data rows
        structured_rows = []

        for tr in rows:
            cells = tr.find_all("td")
            if len(cells) != max_cols:
                continue

            row_obj = {}
            for col, cell in zip(columns, cells):
                key = ".".join(col)
                value = clean_text(cell)

                # numeric normalization
                value = value.replace(",", "")
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass

                row_obj[key] = value

            structured_rows.append(row_obj)

        return {
            "table_context": {
                "section": current_section.get("title"),
                "caption": caption,
                "description": "Structured table extracted from HTML with hierarchical headers",
                "columns": columns,
                "rows": structured_rows
            }
        }

    def _extract_infobox_data(self, table, section: Dict) -> str:
        """
        Extract structured data from a Wikipedia infobox table.

        Returns formatted text with key-value pairs like:
        "Capital: Rome\nLanguages: Latin, Greek\n..."
        """
        infobox_items = []

        # Find all rows in the infobox
        rows = table.find_all("tr")

        for row in rows:
            # Skip header rows (usually the title row)
            header = row.find("th", class_=lambda x: x and "infobox-label" in " ".join(x).lower())
            if not header:
                # Try alternative: look for th elements
                header = row.find("th")

            if header:
                # This is a key-value row
                key = header.get_text(" ", strip=True)

                # Get the value (usually in td)
                value_cell = row.find("td")
                if value_cell:
                    # Extract text, handling links and lists
                    value_parts = []

                    # Handle links in the value
                    for link in value_cell.find_all("a"):
                        href = link.get("href", "")
                        link_text = link.get_text(" ", strip=True)
                        if href.startswith("/wiki/"):
                            link_part = href.split("/wiki/")[1]
                            if ":" not in link_part:
                                title = unquote(link_part).replace("_", " ")
                                section["links"].append(title)
                        value_parts.append(link_text)

                    # If no links found, get all text
                    if not value_parts:
                        value_text = value_cell.get_text(" ", strip=True)
                    else:
                        # Combine link texts, removing duplicates while preserving order
                        seen = set()
                        unique_parts = []
                        for part in value_cell.stripped_strings:
                            if part not in seen:
                                seen.add(part)
                                unique_parts.append(part)
                        value_text = ", ".join(unique_parts)

                    if value_text:
                        infobox_items.append(f"{key}: {value_text}")

            # Extract images from infobox
            for img in row.find_all("img"):
                src = img.get("src") or img.get("data-src")
                if src:
                    # Normalize protocol-relative URLs
                    if src.startswith("//"):
                        src = "https:" + src
                    elif src.startswith("/"):
                        src = "https://en.wikipedia.org" + src
                    elif not src.startswith("http"):
                        src = "https://en.wikipedia.org" + src
                    section["images"].append(src)

        # Format as readable text
        if infobox_items:
            return "\n".join(infobox_items)
        return ""
