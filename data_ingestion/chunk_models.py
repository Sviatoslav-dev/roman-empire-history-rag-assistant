from dataclasses import dataclass, field
from typing import List

from data_ingestion.wikipedia_image import WikipediaImage


@dataclass(slots=True)
class ChunkImageMention:
    """An image mentioned/embedded inside an article chunk."""

    image: WikipediaImage
    caption: str = ""


@dataclass(slots=True)
class ArticleChunk:
    """A single chunk (section/fragment) of an article used for embedding and retrieval."""

    page_title: str
    page_url: str
    section_title: str
    section_path: str
    section_level: int
    text_parts: List[str] = field(default_factory=list)
    images: List[ChunkImageMention] = field(default_factory=list)
    text: str = ""

    EXCLUDED_TITLES = {
        "References",
        "Notes",
        "See also",
        "External links",
        "Citations",
        "Bibliography",
        "Sources",
    }

    def finalize_text(self) -> None:
        if not self.text:
            self.text = "\n\n".join(self.text_parts).strip()

    def is_excluded(self) -> bool:
        return self.section_title in self.EXCLUDED_TITLES

    def is_empty(self) -> bool:
        self.finalize_text()
        return not bool(self.text)

    def chars_count(self) -> int:
        return sum(len(t) for t in self.text_parts)
