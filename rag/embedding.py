"""Text and image embedding models."""
import os

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import open_clip
from PIL import Image
import numpy as np
from typing import List, Union

from logger import get_logger

load_dotenv()

logger = get_logger(__name__)

TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL")
IMAGE_EMBEDDING_MODEL = os.getenv("IMAGE_EMBEDDING_MODEL")

class TextEmbedder:
    """Text embedding model using sentence-transformers."""

    def __init__(self, model_name: str = None):
        """Create the embedder."""
        model_name = model_name or TEXT_EMBEDDING_MODEL
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings


class ImageEmbedder:
    """Image embedding model using OpenCLIP.

    Embeddings are L2-normalized to work well with cosine similarity in Qdrant.
    """

    def __init__(self, model_name: str = None):
        """Create the embedder.

        Args:
            model_name: OpenCLIP model name. Defaults to IMAGE_EMBEDDING_MODEL env var.
        """
        model_name = model_name or IMAGE_EMBEDDING_MODEL
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
    
    def embed(self, image_paths: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for image(s).

        Returns a zero-vector for images that can't be processed (e.g. unsupported
        formats like SVG).
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        embeddings = []
        with torch.no_grad():
            for image_path in image_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    image_features = self.model.encode_image(image_tensor)
                    # Normalize embeddings
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    embeddings.append(image_features.cpu().numpy().flatten())
                except Exception:
                    logger.warning("Failed to process image for embedding: %s", image_path, exc_info=True)
                    # Return zero vector if image can't be processed
                    embeddings.append(np.zeros(512))
        
        return np.array(embeddings)
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s) using OpenCLIP text encoder."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        with torch.no_grad():
            for text in texts:
                text_tokens = self.tokenizer(text).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings.append(text_features.cpu().numpy().flatten())

        return np.array(embeddings)
