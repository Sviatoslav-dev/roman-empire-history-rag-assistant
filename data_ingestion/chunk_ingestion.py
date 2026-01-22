from dataclasses import dataclass, field
from typing import Dict, List
from uuid import uuid4

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
    """Pipeline that owns `ArticleChunk`s and prepares them for Qdrant."""

    loader: WikipediaLoader
    article_filter: WikipediaArticleFilter
    metadata_store: PgMetadataStore
    images_preprocessor: ImagesPreprocessor = field(default_factory=ImagesPreprocessor)

    chunks: List[ArticleChunk] = field(default_factory=list)

    MAX_CHUNK_SIZE: int = 2000  # Max characters per chunk

    def split_articles_into_chunks(self, articles: List[WikipediaArticleScraper]) -> List[ArticleChunk]:
        """Split scraper objects into `ArticleChunk` instances."""
        chunks: List[ArticleChunk] = []
        for article in articles:
            chunks.extend(article.split_by_chunks(self.MAX_CHUNK_SIZE))
        self.chunks = chunks
        logger.info("Total article chunks created: %d", len(chunks))
        return chunks

    def extend_chunks(self, chunks: List[ArticleChunk]) -> None:
        """Append additional chunks to the internal chunk list."""
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
        """Build inputs for upserting TEXT_COLLECTION.

        Returns:
            (chunk_texts, chunk_metadata)

            - chunk_texts: List[str] where each entry is a chunk's finalized text.
            - chunk_metadata: List[dict] aligned 1:1 with chunk_texts. Keys:
                page_title, page_url, section_title, section_path, section_level.

        Notes:
            The Qdrant point id for each text chunk is typically the enumerate() index
            in `self.chunks` at upsert time (unless overridden by caller).
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

    def prepare_images_collection(self) -> tuple[Dict[str, Dict], List[str], List[Dict], List[str]]:
        """Build inputs for upserting IMAGE_COLLECTION.

        Returns:
            (unique_images_by_url, image_paths, image_metadata, image_ids)

            - unique_images_by_url: dict[url -> record] where record contains:
                image_id (str), image_url (str), local_path (str)
            - image_paths: List[str] local paths aligned with image_ids
            - image_metadata: List[dict] aligned with image_paths; keys: image_url, local_path
            - image_ids: List[str] stable ids for this run

        Notes:
            - Unique images are deduplicated by normalized URL.
            - Image local_path is read from PostgreSQL metadata store.
            - We use UUID string point IDs to avoid any possible collision with
              other collections' integer IDs.
        """
        unique_images: Dict[str, Dict] = {}

        for chunk in self.chunks:
            for mention in chunk.images:
                img_url = mention.image.url
                if not img_url or "Blank.png" in img_url:
                    continue

                if img_url in unique_images:
                    continue

                meta = self.metadata_store.get_image_by_url(img_url)
                local_path = meta.local_path if meta is not None else None
                if not local_path:
                    # Image wasn't downloaded / persisted yet, skip.
                    continue

                unique_images[img_url] = {
                    "image_id": str(uuid4()),
                    "image_url": img_url,
                    "local_path": str(local_path),
                }

        image_paths: List[str] = []
        image_metadata: List[Dict] = []
        image_ids: List[str] = []

        for rec in unique_images.values():
            image_ids.append(rec["image_id"])
            image_paths.append(rec["local_path"])
            image_metadata.append({"image_url": rec["image_url"], "local_path": rec["local_path"]})

        return unique_images, image_paths, image_metadata, image_ids

    def prepare_link_collection(self, unique_images_by_url: Dict[str, Dict]) -> List[Dict]:
        """Build relationship payloads for LINK_COLLECTION.

        Args:
            unique_images_by_url: Mapping from image_url -> record {image_id, ...} returned
                by :meth:`prepare_images_collection`.

        Returns:
            List of dict records. Each dict has at minimum:
              - text_chunk_id: int (typically enumerate index of `self.chunks`)
              - image_id: int (from unique_images_by_url)
              - caption: str (may be empty)
              - page_title/page_url/section_title/section_path/section_level

        Notes:
            LINK_COLLECTION records store relationship-level metadata (caption, section),
            not on the image points themselves.
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
