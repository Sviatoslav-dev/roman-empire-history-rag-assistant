from dataclasses import dataclass, field
from typing import Dict, List

from data_ingestion.chunk_models import ArticleChunk
from data_ingestion.images_preprocessor import ImagesPreprocessor
from data_ingestion.pg_metadata_store import PgMetadataStore
from data_ingestion.scraper.wikipedia_article_scraper import WikipediaArticleScraper
from data_ingestion.wikipedia_article_filter import WikipediaArticleFilter
from data_ingestion.wikipedia_loader import WikipediaLoader
from data_ingestion.qdrant_payloads import (
    build_images_collection_payloads,
    build_link_collection_payloads,
    build_text_collection_payloads,
)
from logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ChunkProcessor:
    """Owns `ArticleChunk`s and prepares them for Qdrant."""

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
        return build_text_collection_payloads(self.chunks)

    def prepare_images_collection(self) -> tuple[Dict[str, Dict], List[str], List[Dict], List[str]]:
        """Build inputs for upserting IMAGE_COLLECTION."""

        def _local_path_by_url(url: str) -> str | None:
            meta = self.metadata_store.get_image_by_url(url)
            return str(meta.local_path) if meta is not None and meta.local_path else None

        return build_images_collection_payloads(self.chunks, local_path_by_url=_local_path_by_url)

    def prepare_link_collection(self, unique_images_by_url: Dict[str, Dict]) -> List[Dict]:
        """Build relationship payloads for LINK_COLLECTION."""
        return build_link_collection_payloads(self.chunks, unique_images_by_url)
