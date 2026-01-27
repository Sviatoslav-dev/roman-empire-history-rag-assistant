"""Helpers to build Qdrant upsert payloads from ingestion chunks.

These functions are intentionally small and easy to unit test.
They keep `ChunkProcessor` from accumulating payload-building logic.

Design goals:
- deterministic, side-effect-free payload construction (given dependencies)
- keep DB access out of the pipeline (inject resolver function)
"""

from typing import Callable, Dict, List, Optional
from uuid import uuid4

from data_ingestion.chunk_models import ArticleChunk


def build_text_collection_payloads(chunks: List[ArticleChunk]) -> tuple[List[str], List[Dict]]:
    """Build inputs for upserting TEXT_COLLECTION.

    Returns:
        (chunk_texts, chunk_metadata)
    """
    chunk_texts = [c.text for c in chunks]
    chunk_metadata: List[Dict] = [
        {
            "page_title": chunk.page_title,
            "page_url": chunk.page_url,
            "section_title": chunk.section_title,
            "section_path": chunk.section_path,
            "section_level": chunk.section_level,
        }
        for chunk in chunks
    ]
    return chunk_texts, chunk_metadata


# A tiny protocol via callable: url -> local_path (or None if missing)
LocalPathResolver = Callable[[str], Optional[str]]


def build_images_collection_payloads(
    chunks: List[ArticleChunk],
    *,
    local_path_by_url: LocalPathResolver,
    id_factory: Callable[[], str] = lambda: str(uuid4()),
) -> tuple[Dict[str, Dict], List[str], List[Dict], List[str]]:
    """Build inputs for upserting IMAGE_COLLECTION.

    Args:
        chunks: Article chunks with image mentions.
        local_path_by_url: Callable returning local path for a given image URL.
        id_factory: ID generator for image point IDs (defaults to UUID4 strings).

    Returns:
        (unique_images_by_url, image_paths, image_metadata, image_ids)
    """
    unique_images: Dict[str, Dict] = {}

    for chunk in chunks:
        for mention in chunk.images:
            img_url = mention.image.url
            if not img_url or "Blank.png" in img_url:
                continue

            if img_url in unique_images:
                continue

            local_path = local_path_by_url(img_url)
            if not local_path:
                # Image wasn't downloaded / persisted yet, skip.
                continue

            unique_images[img_url] = {
                "image_id": id_factory(),
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


def build_link_collection_payloads(chunks: List[ArticleChunk], unique_images_by_url: Dict[str, Dict]) -> List[Dict]:
    """Build relationship payloads for LINK_COLLECTION."""
    links: List[Dict] = []

    for chunk_id, chunk in enumerate(chunks):
        for mention in chunk.images:
            img_url = mention.image.url
            if not img_url or "Blank.png" in img_url:
                continue

            rec = unique_images_by_url.get(img_url)
            if not rec:
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

