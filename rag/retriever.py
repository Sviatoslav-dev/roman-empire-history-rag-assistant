"""Vector database retriever."""
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Tuple, Optional

from rag.embedding import TextEmbedder, ImageEmbedder

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
TEXT_COLLECTION_NAME = os.getenv("TEXT_COLLECTION_NAME")
IMAGE_COLLECTION_NAME = os.getenv("IMAGE_COLLECTION_NAME")

class QdrantRetriever:
    """Qdrant-based retriever for text and images."""
    
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
        """Create collections if they don't exist."""
        # Text collection (384 dimensions for all-MiniLM-L6-v2)
        try:
            self.client.get_collection(TEXT_COLLECTION_NAME)
        except Exception:
            self.client.create_collection(
                collection_name=TEXT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
        
        # Image collection (512 dimensions for OpenCLIP ViT-B-32)
        try:
            self.client.get_collection(IMAGE_COLLECTION_NAME)
        except Exception:
            self.client.create_collection(
                collection_name=IMAGE_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=512,
                    distance=Distance.COSINE
                )
            )
    
    def search_text(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Tuple[str, float, dict]]:
        """Search for relevant text chunks."""
        query_vector = self.text_embedder.embed(query)[0]
        
        results = self.client.search(
            collection_name=TEXT_COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold
        )
        
        return [
            (
                point.payload.get("text", ""),
                point.score,
                {k: v for k, v in point.payload.items() if k != "text"}
            )
            for point in results
        ]
    
    def search_images(
        self,
        query_image_path: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Tuple[dict, float]]:
        """Search for similar images."""
        query_vector = self.image_embedder.embed(query_image_path)[0]
        
        results = self.client.search(
            collection_name=IMAGE_COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold
        )
        
        return [
            (point.payload, point.score)
            for point in results
        ]
    
    def search_text_by_text(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, dict]]:
        """Search text using text query (for text-based image search)."""
        return self.search_text(query, top_k)
    
    def search_images_by_text(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3
    ) -> List[Tuple[dict, float]]:
        """Search for images using a text query by embedding the text with OpenCLIP."""
        # Use OpenCLIP's text encoder to embed the query (same space as images)
        query_vector = self.image_embedder.embed_text(query)[0]
        
        # Search images collection using the text embedding
        results = self.client.search(
            collection_name=IMAGE_COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold
        )
        
        return [
            (point.payload, point.score)
            for point in results
        ]
    
    def add_text_chunks(
        self,
        texts: List[str],
        metadata: List[dict],
        ids: Optional[List[int]] = None
    ):
        """Add text chunks to the collection."""
        # embeddings = self.text_embedder.embed(texts)
        #
        # points = [
        #     PointStruct(
        #         id=ids[i] if ids else i,
        #         vector=embeddings[i].tolist(),
        #         payload={
        #             "text": texts[i],
        #             **metadata[i]
        #         }
        #     )
        #     for i in range(len(texts))
        # ]
        #
        # # try:
        # self.client.upsert(
        #     collection_name=TEXT_COLLECTION_NAME,
        #     points=points
        # )
        # # except Exception as e:
        # #     print(f"Error upserting text chunks: {e}")
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
        """Add images to the collection."""
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
