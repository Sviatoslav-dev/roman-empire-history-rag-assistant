import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from typing import List, Tuple, Optional

from rag.embedding import TextEmbedder, ImageEmbedder

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
TEXT_COLLECTION_NAME = os.getenv("TEXT_COLLECTION_NAME")
IMAGE_COLLECTION_NAME = os.getenv("IMAGE_COLLECTION_NAME")
LINK_COLLECTION_NAME = os.getenv("LINK_COLLECTION_NAME", "chunk_image_links")


def _is_missing_collection_error(exc: UnexpectedResponse) -> bool:
    """Return True if the exception indicates Qdrant collection is missing (HTTP 404)."""
    status = getattr(exc, "status_code", None)
    if status == 404:
        return True

    # Some versions expose a `response` object.
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 404:
        return True

    return False


class QdrantRetriever:
    """Retrieve text chunks and images from Qdrant.

    Collections and IDs:
    - Text points are addressed by `chunk_id` (Qdrant point id).
    - Image points are addressed by `image_id` (Qdrant point id).
    - Link points are arbitrary; their payload ties together `text_chunk_id` and `image_id`.
    """

    def __init__(self):
        """Initialize Qdrant client and collections."""
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT
        )
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()

        # Ensure collections exist
        self._ensure_collections()

    def _ensure_collections(self):
        """Create required Qdrant collections if they don't exist."""
        # Text collection (384 dimensions for all-MiniLM-L6-v2)
        try:
            self.client.get_collection(TEXT_COLLECTION_NAME)
        except UnexpectedResponse as e:
            if not _is_missing_collection_error(e):
                raise
            self.client.create_collection(
                collection_name=TEXT_COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        # Image collection (512 dimensions for OpenCLIP ViT-B-32)
        try:
            self.client.get_collection(IMAGE_COLLECTION_NAME)
        except UnexpectedResponse as e:
            if not _is_missing_collection_error(e):
                raise
            self.client.create_collection(
                collection_name=IMAGE_COLLECTION_NAME,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )

        # Link collection: dummy 1D vectors, used only for payload filtering.
        # Qdrant requires vectors unless you use sparse-only collections; we keep this simple.
        try:
            self.client.get_collection(LINK_COLLECTION_NAME)
        except UnexpectedResponse as e:
            if not _is_missing_collection_error(e):
                raise
            self.client.create_collection(
                collection_name=LINK_COLLECTION_NAME,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Tuple[str, float, dict, int]]:
        """Search for relevant text chunks.
        """
        query_vector = self.text_embedder.embed(query)[0]

        results = self.client.query_points(
            collection_name=TEXT_COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold
        )

        return [
            (
                point.payload.get("text", ""),
                point.score,
                {k: v for k, v in point.payload.items() if k != "text"},
                point.id  # Add chunk ID for linking to images
            )
            for point in results.points
        ]

    def search_images(
        self,
        query_image_path: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Tuple[dict, float]]:
        """Search for similar images using an image file as a query.

        Args:
            query_image_path: Local image path to embed and search by.
            top_k: Maximum number of points to return.
            score_threshold: Qdrant score threshold.

        Returns:
            List of tuples (payload, score).

            The returned payload is always augmented with `image_id` equal to the
            Qdrant point id (if not already present in payload).
        """
        query_vector = self.image_embedder.embed(query_image_path)[0]

        results = self.client.query_points(
            collection_name=IMAGE_COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )

        out: List[Tuple[dict, float]] = []
        for point in results.points:
            payload = dict(point.payload or {})
            # If ingestion didn't store image_id in payload, use point.id as the stable id.
            payload.setdefault("image_id", point.id)
            out.append((payload, point.score))

        return out

    def search_text_by_text(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, dict, int]]:
        """Search text using text query."""
        return self.search_text(query, top_k)

    def search_images_by_text(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3
    ) -> List[Tuple[dict, float]]:
        """Search images by embedding the *text query*.

        Returns:
            List of (payload, score) where payload includes `image_id`.
        """
        query_vector = self.image_embedder.embed_text(query)[0]

        results = self.client.query_points(
            collection_name=IMAGE_COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )

        out: List[Tuple[dict, float]] = []
        for point in results.points:
            payload = dict(point.payload or {})
            payload.setdefault("image_id", point.id)
            out.append((payload, point.score))

        return out

    def get_images_by_text_chunk_id(
        self,
        text_chunk_id: int
    ) -> List[dict]:
        """Return images linked to a given text chunk.

        This is a two-step lookup:
        1) Filter LINK_COLLECTION by payload.text_chunk_id
        2) Retrieve image points by the linked payload.image_id values

        The returned list contains image payload dicts augmented with relationship
        fields copied from the corresponding link payload when present:
        - caption
        - section_title/section_path/section_level
        - page_title/page_url

        Args:
            text_chunk_id: Text chunk point id in TEXT_COLLECTION.

        Returns:
            List of image payload dictionaries.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        link_points, _ = self.client.scroll(
            collection_name=LINK_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="text_chunk_id", match=MatchValue(value=text_chunk_id))
                ]
            ),
            limit=500,
            with_payload=True,
            with_vectors=False
        )

        if not link_points:
            return []

        # Collect unique image ids
        image_ids = []
        link_by_image_id = {}
        for p in link_points:
            payload = p.payload or {}
            image_id = payload.get("image_id")
            if image_id is None:
                continue
            if image_id not in link_by_image_id:
                link_by_image_id[image_id] = payload
                image_ids.append(image_id)

        if not image_ids:
            return []

        image_points = self.client.retrieve(
            collection_name=IMAGE_COLLECTION_NAME,
            ids=image_ids,
            with_payload=True,
            with_vectors=False
        )

        out: List[dict] = []
        for img_point in image_points or []:
            img_payload = dict(img_point.payload or {})
            link_payload = link_by_image_id.get(img_point.id, {})
            # link-specific info should not live on the image point
            for k in ("caption", "section_title", "section_path", "section_level", "page_title", "page_url", "text_chunk_id"):
                if k in link_payload and link_payload.get(k) is not None:
                    img_payload[k] = link_payload.get(k)
            out.append(img_payload)

        return out

    def get_text_chunk_ids_by_image_id(self, image_id: int) -> List[int]:
        """Return all text chunk ids linked to a given image id."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        link_points, _ = self.client.scroll(
            collection_name=LINK_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="image_id", match=MatchValue(value=image_id))
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        ids = []
        for p in link_points or []:
            cid = (p.payload or {}).get("text_chunk_id")
            if cid is not None:
                ids.append(cid)
        return ids

    def add_chunk_image_links(
        self,
        links: List[dict],
        ids: Optional[List[int]] = None
    ):
        """Upsert chunk-image link dicts into LINK_COLLECTION.

        Link payload schema (minimum):
            {
              'text_chunk_id': int,   # TEXT_COLLECTION point id
              'image_id': int,        # IMAGE_COLLECTION point id
              'caption': str,         # optional; relationship-level
              'page_title': str,
              'page_url': str,
              'section_title': str,
              'section_path': str,
              'section_level': int,
            }

        Args:
            links: List of link payload dicts.
            ids: Optional explicit point ids for link points (defaults to enumerate()).
        """
        points = [
            PointStruct(
                id=ids[i] if ids else i,
                vector=[1.0],
                payload=links[i]
            )
            for i in range(len(links))
        ]

        self.client.upsert(
            collection_name=LINK_COLLECTION_NAME,
            points=points
        )

    def add_text_chunks(
        self,
        texts: List[str],
        metadata: List[dict],
        ids: Optional[List[int]] = None
    ):
        """Upsert text chunks into TEXT_COLLECTION.

        Args:
            texts: List of chunk texts.
            metadata: List of metadata dicts aligned 1:1 with texts.
            ids: Optional explicit Qdrant point ids.

        Notes:
            Performs simple batching with retry/backoff.
        """
        embeddings = self.text_embedder.embed(texts)

        points = [
            PointStruct(
                id=ids[i] if ids else i,
                vector=embeddings[i].tolist(),
                payload={
                    "text": texts[i],
                    **metadata[i]
                }
            )
            for i in range(len(texts))
        ]

        # Upsert in batches with simple retry/backoff
        from time import sleep

        chunk_size = 64
        max_retries = 3

        for start in range(0, len(points), chunk_size):
            batch = points[start:start + chunk_size]
            for attempt in range(1, max_retries + 1):
                try:
                    self.client.upsert(
                        collection_name=TEXT_COLLECTION_NAME,
                        points=batch
                    )
                    break
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    sleep(2 ** (attempt - 1))

    def add_images(
        self,
        image_paths: List[str],
        metadata: List[dict],
        ids: Optional[List[int]] = None
    ):
        """Upsert images into IMAGE_COLLECTION.

        Args:
            image_paths: Local paths to images.
            metadata: List of metadata dicts aligned 1:1 with image_paths.
            ids: Optional explicit Qdrant point ids.
        """
        embeddings = self.image_embedder.embed(image_paths)

        points = [
            PointStruct(
                id=ids[i] if ids else i,
                vector=embeddings[i].tolist(),
                payload={
                    "image_path": image_paths[i],
                    **metadata[i]
                }
            )
            for i in range(len(image_paths))
        ]

        self.client.upsert(
            collection_name=IMAGE_COLLECTION_NAME,
            points=points
        )

    def get_text_chunk_by_id(
        self,
        chunk_id: int
    ) -> Optional[Tuple[str, dict]]:
        """Return a single text chunk by id.

        Returns:
            (text, metadata) or None if the point doesn't exist.
        """
        points = self.client.retrieve(
            collection_name=TEXT_COLLECTION_NAME,
            ids=[chunk_id],
            with_payload=True,
            with_vectors=False
        )
        if points and len(points) > 0:
            payload = points[0].payload or {}
            return payload.get("text", ""), {k: v for k, v in payload.items() if k != "text"}
        return None

    def get_links_by_image_id(self, image_id: str | int, limit: int = 50) -> List[dict]:
        """Return link payload dicts for a given image id.

        Link payloads contain relationship-level fields such as caption and
        text_chunk_id.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        link_points, _ = self.client.scroll(
            collection_name=LINK_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="image_id", match=MatchValue(value=image_id))
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        return [p.payload or {} for p in (link_points or [])]
