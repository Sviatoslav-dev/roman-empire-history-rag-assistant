from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from data_ingestion.chunk_models import ArticleChunk
from data_ingestion.images_preprocessor import ImagesPreprocessor
from data_ingestion.pg_metadata_store import PgMetadataStore
from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.wikipedia_article_filter import WikipediaArticleFilter
from data_ingestion.wikipedia_loader import WikipediaLoader
from logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ChunkIngestionPipeline:
    """Pipeline that owns `ArticleChunk`s and prepares them for Qdrant.

    Responsibilities:
    - keep chunks as state (`self.chunks`)
    - download images referenced by chunks
    - post-process downloaded images (SVG -> PNG)
    - filter chunk image mentions by license
    - build Qdrant payloads: text chunks, unique images, and chunk-image links

    Notes:
    - This class does not talk to Qdrant directly; it only prepares payloads.
    - Qdrant payloads stay dict-based (storage boundary).
    """

    loader: WikipediaLoader
    article_filter: WikipediaArticleFilter
    metadata_store: PgMetadataStore
    images_preprocessor: ImagesPreprocessor = field(default_factory=ImagesPreprocessor)

    chunks: List[ArticleChunk] = field(default_factory=list)

    IMAGE_ID_START: int = 1_000_000

    # def set_chunks(self, chunks: List[ArticleChunk]) -> None:
    #     self.chunks = list(chunks)

    def split_articles_into_chunks(self, articles: List[WikipediaArticleScraper]) -> List[ArticleChunk]:
        """Split scraper objects into `ArticleChunk` instances."""
        chunks: List[ArticleChunk] = []
        for article in articles:
            chunks.extend(article.split_by_chunks(2000))
        self.chunks = chunks
        logger.info("Total article chunks created: %d", len(chunks))
        return chunks


    def extend_chunks(self, chunks: List[ArticleChunk]) -> None:
        self.chunks.extend(chunks)

    def download_images(self) -> None:
        """Download all images referenced by current chunks."""
        images = [m.image for ch in self.chunks for m in ch.images]
        logger.info("Downloading %d image mentions...", len(images))
        self.loader.download_images(images)

    def postprocess_images(self) -> None:
        """Post-process downloaded images (currently: convert SVG to PNG)."""
        self.images_preprocessor.convert_svgs_to_png()

    def filter_images_by_license(self) -> None:
        """Filter image mentions inside chunks using the metadata store license."""
        self.chunks = self.article_filter.filter_chunk_images(self.chunks)

    def prepare_text_collection(self) -> tuple[List[str], List[Dict]]:
        """Prepare payload for the TEXT collection.

        Returns:
            (chunk_texts, chunk_metadata)
        """
        chunk_texts = [c.text for c in self.chunks]
        chunk_metadata: List[Dict] = [
            {
                "page_title": chunk.page_title,
                "page_url": chunk.page_url,
                "section_title": chunk.section_title,
                "section_path": chunk.section_path,
                "section_level": chunk.section_level,
            }
            for chunk in self.chunks
        ]
        return chunk_texts, chunk_metadata

    def prepare_images_collection(self) -> tuple[Dict[str, Dict], List[str], List[Dict], List[int]]:
        """Prepare payload for the IMAGE collection.

        Returns:
            (unique_images_by_url, image_paths, image_metadata, image_ids)

        Notes:
            - Unique images are deduplicated by normalized URL.
            - IDs are deterministic within this run, starting from IMAGE_ID_START.
        """
        unique_images: Dict[str, Dict] = {}
        next_image_id = self.IMAGE_ID_START

        for chunk in self.chunks:
            for mention in chunk.images:
                img_url = mention.image.url
                if not img_url or "Blank.png" in img_url:
                    continue

                if img_url in unique_images:
                    continue

                local_path = self.metadata_store.get_image_by_url(img_url).local_path
                unique_images[img_url] = {
                    "image_id": next_image_id,
                    "image_url": img_url,
                    "local_path": str(local_path),
                }
                next_image_id += 1

        image_paths: List[str] = []
        image_metadata: List[Dict] = []
        image_ids: List[int] = []

        for rec in unique_images.values():
            image_ids.append(rec["image_id"])
            image_paths.append(rec["local_path"])
            image_metadata.append({"image_url": rec["image_url"], "local_path": rec["local_path"]})

        return unique_images, image_paths, image_metadata, image_ids

    def prepare_link_collection(self, unique_images_by_url: Dict[str, Dict]) -> List[Dict]:
        """Prepare payload for the LINK collection.

        Args:
            unique_images_by_url: Mapping from image_url -> {image_id, ...}.

        Returns:
            List of link records connecting text_chunk_id to image_id.
        """
        links: List[Dict] = []

        for chunk_id, chunk in enumerate(self.chunks):
            for mention in chunk.images:
                img_url = mention.image.url
                if not img_url or "Blank.png" in img_url:
                    continue

                rec = unique_images_by_url.get(img_url)
                if not rec:
                    # Image mention filtered out (e.g. missing metadata/license), skip.
                    continue

                caption = (mention.caption or "").strip()

                links.append(
                    {
                        "text_chunk_id": chunk_id,
                        "image_id": rec["image_id"],
                        "caption": caption,
                        "page_title": chunk.page_title,
                        "page_url": chunk.page_url,
                        "section_title": chunk.section_title,
                        "section_path": chunk.section_path,
                        "section_level": chunk.section_level,
                    }
                )

        return links

