from __future__ import annotations

import json
from typing import Optional, List

from bs4 import Tag

from data_ingestion.scraper.base_page_scraper import BasePageScraper
from data_ingestion.wikipedia_image import WikipediaImage
from data_ingestion.chunk_models import ArticleChunk, ChunkImageMention
from logger import get_logger

logger = get_logger(__name__)


class WikipediaArticleScraper(BasePageScraper):
    """Parses and analyzes a single Wikipedia article HTML."""

    PROBLEM_KEYWORDS = frozenset([
        "disput", "cleanup", "update", "problem", "outdat",
        "bias", "needs sources", "merge", "contradictory"
    ])

    BANNER_SELECTORS = frozenset([
        ".ambox", ".hatnote", ".mw-warning", ".metadata", ".notice", ".messagebox"
    ])


    def passes_quality_filters(self, min_article_length: int, min_citations_number: int) -> bool:
        """
        Check whether the article meets basic quality requirements.

        The article is considered valid if it:
        - contains sufficient visible text (more than min_article_length characters),
        - has an adequate number of citations (min_citations_number or more),
        - does not include maintenance/problem/update banners,
        - represents an English Wikipedia article.

        Args:
            min_article_length: Minimum number of visible characters required.
            min_citations_number: Minimum number of unique citation anchors.

        Returns:
            True if the article passes all quality filters, False otherwise.
        """
        # Length check: use visible text inside content area
        visible_text = self._get_visible_text()
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
        """Estimate the number of unique inline citations in the article."""
        refs = set()
        for a in self.soup.select("a[href^='#cite_note']"):
            refs.add(a["href"])
        return len(refs)

    def has_problem_or_update_box(self) -> bool:
        """Detect whether the page has maintenance/problem/update/dispute banners."""
        for sel in self.BANNER_SELECTORS:
            for el in self.soup.select(sel):
                text = el.get_text()
                normalized = ' '.join(text.split()).lower()
                if any(kw in normalized for kw in self.PROBLEM_KEYWORDS):
                    return True

        # Also check for templates rendered as tables with class 'ambox'
        for el in self.soup.select("table.ambox, div.ambox"):
            text = ' '.join(el.get_text().split()).lower()
            if any(keyword in text for keyword in self.PROBLEM_KEYWORDS):
                return True

        # Check top of content for maintenance phrases
        content = self.soup.select_one("#mw-content-text")
        if content:
            paragraphs = content.select("p")[:2]
            text_parts = [' '.join(p.get_text().split()) for p in paragraphs]
            top_text = ' '.join(text_parts).lower()

            if any(keyword in top_text for keyword in self.PROBLEM_KEYWORDS):
                return True

        return False

    def is_english_article(self) -> bool:
        """Check if the page HTML is English.

        Returns:
            True when the HTML language is English or the content looks like an
            English Wikipedia article; False otherwise.
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

    def _get_visible_text(self) -> str:
        content_el = self.soup.select_one("#mw-content-text")
        return " ".join(content_el.get_text().split()) if content_el else " ".join(self.soup.get_text().split())

    def texts_chars_count(self, texts: List[str]) -> int:
        texts_len = [len(text) for text in texts]
        return sum(texts_len)

    def _content_elements(self, parent):
        search_tags = ["h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "div", "figure", "meta", "link"]

        elements = []

        content_elements = parent.find_all(search_tags, recursive=False)

        for el in content_elements:
            if el.name in ("meta", "link"):
                content_elements.extend(self._content_elements(el))
            else:
                elements.append(el)
        return elements

    def split_by_chunks(self, max_text_size: int = 2000) -> List[ArticleChunk]:
        """Split the article into heading-based chunks.

        A chunk roughly corresponds to a section (h2–h6) and is further split when its
        accumulated text exceeds `max_text_size`.

        Args:
            max_text_size: Maximum combined size of `text_parts` (characters). If exceeded,
                the current chunk is finalized/emitted and a continuation chunk is started
                with the same section metadata.

        Returns:
            A list of `ArticleChunk` objects.
        """
        self._cleanup_for_chunking()

        content_root = self._get_content_root()
        content_elements = self._content_elements(content_root)

        chunks: List[ArticleChunk] = []
        heading_stack: List[tuple[int, str]] = []

        current_chunk: Optional[ArticleChunk] = self._new_chunk(
            section_title="Introduction",
            section_path="Introduction",
            section_level=1,
        )

        for el in content_elements:
            if not getattr(el, "name", None):
                continue

            # A) Section heading
            if (next_chunk := self._maybe_start_new_section(el, heading_stack, current_chunk, chunks)) is not None:
                current_chunk = next_chunk
                continue

            if current_chunk is None:
                continue

            # stop once we hit excluded tail sections
            if current_chunk.is_excluded():
                logger.debug(
                    "Stopping chunking at excluded section '%s' for article '%s'",
                    current_chunk.section_title,
                    self.title,
                )
                break

            current_chunk = self._handle_content_element(el, current_chunk, chunks, max_text_size=max_text_size)

        self._finalize_and_append(current_chunk, chunks)

        logger.info("Chunked '%s' into %d chunks", self.title, len(chunks))
        return chunks

    # -------------------------
    # Helpers (chunking)
    # -------------------------

    def _cleanup_for_chunking(self) -> None:
        """Remove noisy navigation/boilerplate blocks that should not be embedded."""
        toc = self.soup.find("div", id="toc")
        if toc:
            toc.decompose()

        for tag in self.soup.find_all(attrs={"role": ["navigation", "presentation"]}):
            tag.decompose()

        for tag in self.soup.find_all(class_="side-box"):
            tag.decompose()

    def _get_content_root(self) -> Tag:
        """Return the root element that contains the visible article content."""
        content_root = self.soup.find("div", class_="mw-content-ltr")
        if content_root is None:
            raise Exception("Could not find content root in HTML")
        return content_root

    def _new_chunk(self, *, section_title: str, section_path: str, section_level: int) -> ArticleChunk:
        """Factory for a new chunk keeping page-level metadata consistent."""
        return ArticleChunk(
            page_title=self.title,
            page_url=self.url,
            section_title=section_title,
            section_path=section_path,
            section_level=section_level,
        )

    def _finalize_and_append(self, chunk: Optional[ArticleChunk], chunks: List[ArticleChunk]) -> None:
        """Finalize a chunk and append it if it is non-empty and not excluded."""
        if chunk is None:
            return

        chunk.finalize_text()
        if chunk.is_empty() or chunk.is_excluded():
            return

        chunks.append(chunk)

    def _maybe_start_new_section(
        self,
        el: Tag,
        heading_stack: List[tuple[int, str]],
        current_chunk: Optional[ArticleChunk],
        chunks: List[ArticleChunk],
    ) -> Optional[ArticleChunk]:
        """If `el` contains a heading, close current chunk and start a new one."""
        h = el.find(["h2", "h3", "h4", "h5", "h6"])
        if not h:
            return None

        level = int(h.name[1])
        title_text = " ".join(h.get_text().split())
        if not title_text:
            return None

        # Maintain hierarchy stack
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title_text))
        title_path = " > ".join(t for _, t in heading_stack)

        # flush previous
        self._finalize_and_append(current_chunk, chunks)

        logger.debug("Start section '%s' (level=%d) for '%s'", title_text, level, self.title)

        return self._new_chunk(section_title=title_text, section_path=title_path, section_level=level)

    def _handle_content_element(
        self,
        el: Tag,
        current_chunk: ArticleChunk,
        chunks: List[ArticleChunk],
        *,
        max_text_size: int,
    ) -> ArticleChunk:
        """Handle one content element and update/return the current chunk."""
        classes_attr = el.get("class")
        el_classes = " ".join(classes_attr) if isinstance(classes_attr, list) else (classes_attr or "")
        el_classes = el_classes.lower()

        # Split chunks by figure elements
        if el.name == "figure" and not current_chunk.text_parts:
            self._finalize_and_append(current_chunk, chunks)
            current_chunk = self._new_chunk(
                section_title=current_chunk.section_title,
                section_path=current_chunk.section_path,
                section_level=current_chunk.section_level,
            )

        # Tables
        if el.name == "table":
            self._handle_table(el, current_chunk)
            return current_chunk

        # Figure images
        if el.name == "figure":
            self._handle_figure(el, current_chunk)
            return current_chunk

        # Thumbnails
        if "thumb" in el_classes:
            self._handle_thumbnails(el, current_chunk)
            return current_chunk

        # Galleries
        if "gallery" in el_classes:
            self._handle_gallery(el, current_chunk)
            return current_chunk

        # Text content
        if el.name in {"p", "ul", "ol", "table"}:
            text = " ".join(el.get_text().split())
            if text:
                current_chunk.text_parts.append(text)

                if current_chunk.chars_count() > max_text_size:
                    self._finalize_and_append(current_chunk, chunks)
                    current_chunk = self._new_chunk(
                        section_title=current_chunk.section_title,
                        section_path=current_chunk.section_path,
                        section_level=current_chunk.section_level,
                    )

        return current_chunk

    def _handle_table(self, el: Tag, current_chunk: ArticleChunk) -> None:
        """Extract infobox or generic table JSON and append it into the chunk text."""
        classes_attr = el.get("class")
        class_str = " ".join(classes_attr) if isinstance(classes_attr, list) else (classes_attr or "")

        if "infobox" in class_str.lower():
            infobox_text = self._extract_infobox_data(el, current_chunk)
            if infobox_text:
                current_chunk.text_parts.append(infobox_text)
                logger.debug("Infobox extracted for '%s' / section '%s'", self.title, current_chunk.section_path)
            return

        table_json = self._extract_table_generic_json(el, current_chunk)
        if table_json:
            current_chunk.text_parts.append(json.dumps(table_json, ensure_ascii=False))

    def _handle_figure(self, el: Tag, current_chunk: ArticleChunk) -> None:
        """Extract a <figure> image and caption and attach it to the chunk."""
        img: Optional[Tag] = el.select_one("a img")
        if not img:
            return

        src = self._get_image_url(img)
        if not src:
            return

        caption_el: Optional[Tag] = el.select_one("figcaption")
        caption = " ".join(caption_el.get_text().split()) if caption_el else ""

        wiki_image = WikipediaImage(src)
        wiki_image.normalize_url()
        current_chunk.images.append(ChunkImageMention(image=wiki_image, caption=caption))

        logger.debug("Figure image captured: %s (caption_len=%d)", wiki_image.url, len(caption))

    def _handle_thumbnails(self, el: Tag, current_chunk: ArticleChunk) -> None:
        """Extract thumbnail images in a 'thumb' container."""
        for ts in el.select(".tsingle"):
            img: Optional[Tag] = ts.select_one("a img")
            if not img:
                continue

            src = self._get_image_url(img)
            if not src:
                continue

            thumbcaption = ts.select_one(".thumbcaption")
            if thumbcaption:
                caption = " ".join(thumbcaption.get_text().split())
            else:
                cap_el = el.select_one(".thumbcaption")
                caption = " ".join(cap_el.get_text().split()) if cap_el else ""

            wiki_image = WikipediaImage(src)
            wiki_image.normalize_url()
            current_chunk.images.append(ChunkImageMention(image=wiki_image, caption=caption))

    def _handle_gallery(self, el: Tag, current_chunk: ArticleChunk) -> None:
        """Extract images from a Wikipedia gallery block."""
        for gallery_item in el.select(".gallerybox"):
            img: Optional[Tag] = gallery_item.select_one("a img")
            if not img:
                continue

            src = self._get_image_url(img)
            if not src:
                continue

            caption_el = gallery_item.select_one(".gallerytext")
            caption = " ".join(caption_el.get_text().split()) if caption_el else ""

            wiki_image = WikipediaImage(src)
            wiki_image.normalize_url()
            current_chunk.images.append(ChunkImageMention(image=wiki_image, caption=caption))

    def _get_image_url(self, img: Tag) -> Optional[str]:
        """Extract and normalize an image URL from an <img> tag."""
        src_val = img.get("src") or img.get("data-src") or img.get("href")

        if isinstance(src_val, list):
            src = src_val[0] if src_val else None
        else:
            src = src_val

        if not isinstance(src, str) or not src:
            return None

        # Skip data URIs and very small images (likely icons)
        if src.startswith("data:") or "icon" in src.lower():
            return None

        # Normalize protocol-relative URLs
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = "https://en.wikipedia.org" + src
        elif not src.startswith("http"):
            src = "https://en.wikipedia.org" + src

        return src

    def _extract_table_generic_json(self, table, current_chunk: ArticleChunk):
        """Extract a generic HTML table into a structured JSON-like dict.

        Output is appended as JSON text into the chunk and is intentionally storage-friendly.

        The returned schema is kept stable because downstream uses store it as plain text.
        """
        caption, rows = self._table_extract_caption_and_rows(table)
        if not rows:
            return None

        header_rows, data_rows = self._table_split_header_and_data(rows)
        columns = self._table_build_column_hierarchies(header_rows)
        if not columns:
            return None

        structured_rows = self._table_parse_data_rows(data_rows, columns)

        logger.debug(
            "Parsed table in '%s' / section '%s': cols=%d rows=%d (caption=%s)",
            self.title,
            current_chunk.section_path,
            len(columns),
            len(structured_rows),
            bool(caption),
        )

        return {
            "table_context": {
                "section": current_chunk.section_title,
                "caption": caption,
                "description": "Structured table extracted from HTML with hierarchical headers",
                "columns": columns,
                "rows": structured_rows,
            }
        }

    def _table_extract_caption_and_rows(self, table) -> tuple[Optional[str], list]:
        """Return (caption_text, rows) where rows is a list of <tr> tags."""
        def clean_text(el) -> str:
            return " ".join(getattr(el, "stripped_strings", []) or [])

        rows = table.find_all("tr")
        if not rows:
            return None, []

        caption = None
        first_row_cells = rows[0].find_all(["td", "th"])
        if len(first_row_cells) == 1 and first_row_cells[0].has_attr("colspan"):
            caption = clean_text(first_row_cells[0])
            rows = rows[1:]

        return caption, rows

    def _table_split_header_and_data(self, rows: list) -> tuple[list, list]:
        """Split table rows into header rows (leading rows containing <th>) and data rows."""
        header_rows = []
        while rows and rows[0].find_all("th"):
            header_rows.append(rows.pop(0))
        return header_rows, rows

    def _table_build_column_hierarchies(self, header_rows: list) -> list[list[str]]:
        """Build column hierarchy arrays from multi-row headers."""
        def clean_text(el) -> str:
            return " ".join(getattr(el, "stripped_strings", []) or [])

        if not header_rows:
            return []

        header_matrix: list[list[str]] = []
        max_cols = 0

        for hr in header_rows:
            row: list[str] = []
            for cell in hr.find_all("th"):
                text = clean_text(cell)
                colspan = int(cell.get("colspan", 1))
                row.extend([text] * colspan)
            max_cols = max(max_cols, len(row))
            header_matrix.append(row)

        # normalize all header rows length
        for r in header_matrix:
            if len(r) < max_cols:
                r.extend([""] * (max_cols - len(r)))

        columns: list[list[str]] = []
        for col_idx in range(max_cols):
            hierarchy: list[str] = []
            for r in header_matrix:
                if r[col_idx]:
                    hierarchy.append(r[col_idx])
            columns.append(hierarchy)

        return columns

    def _table_parse_data_rows(self, data_rows: list, columns: list[list[str]]) -> list[dict]:
        """Parse data rows into list of dicts keyed by dot-joined header hierarchy."""
        def clean_text(el) -> str:
            return " ".join(getattr(el, "stripped_strings", []) or [])

        max_cols = len(columns)
        structured_rows: list[dict] = []

        for tr in data_rows:
            cells = tr.find_all("td")
            if len(cells) != max_cols:
                continue

            row_obj: dict = {}
            for col, cell in zip(columns, cells):
                key = ".".join(col)
                value: object = clean_text(cell)

                # numeric normalization
                if isinstance(value, str):
                    raw = value.replace(",", "")
                    try:
                        value = float(raw) if "." in raw else int(raw)
                    except ValueError:
                        value = value

                row_obj[key] = value

            structured_rows.append(row_obj)

        return structured_rows

    def _extract_infobox_data(self, table, section: ArticleChunk) -> str:
        """Extract a Wikipedia infobox into readable key-value lines.

        Also collects images found inside the infobox and attaches them to `section.images`.

        Returns:
            A newline-delimited text representation, or empty string if nothing found.
        """
        rows = table.find_all("tr")
        if not rows:
            return ""

        items: list[str] = []
        images_added = 0

        for row in rows:
            key_value = self._infobox_extract_key_value(row)
            if key_value:
                items.append(key_value)

            if self._infobox_maybe_extract_image(row, section):
                images_added += 1

        if images_added:
            logger.debug(
                "Infobox images added for '%s' / section '%s': %d",
                self.title,
                section.section_path,
                images_added,
            )

        return "\n".join(items) if items else ""

    def _infobox_extract_key_value(self, row) -> Optional[str]:
        """Extract a single 'Key: Value' line from an infobox row."""
        header = row.find("th", class_=lambda x: x and "infobox-label" in " ".join(x).lower())
        if not header:
            header = row.find("th")
        if not header:
            return None

        key = " ".join(header.get_text().split())
        if not key:
            return None

        value_cell = row.find("td")
        if not value_cell:
            return None

        value_text = " ".join(value_cell.get_text().split())
        if not value_text:
            return None

        return f"{key}: {value_text}"

    def _infobox_maybe_extract_image(self, row, section: ArticleChunk) -> bool:
        """Extract an <img> inside an infobox row, if present."""
        img = row.select_one("img")
        if not img:
            return False

        src = self._get_image_url(img)
        if not src:
            return False

        wiki_image = WikipediaImage(src)
        wiki_image.normalize_url()

        caption = " ".join(row.get_text().split())
        section.images.append(ChunkImageMention(image=wiki_image, caption=caption))
        return True
