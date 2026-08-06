"""
Embedder - Encodes passages and queries into dense vector embeddings using SentenceTransformers (multilingual-e5-small).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import threading
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.core.config import get_config

logger = logging.getLogger(__name__)


class Embedder:
    """Encodes passages and queries into dense vector embeddings using SentenceTransformers (intfloat/multilingual-e5-small)."""

    def __init__(self, model_name: str = None, device: str = None):
        cfg = get_config()
        self.model_name = model_name or cfg.get("embedding.model_name", "intfloat/multilingual-e5-small")
        self.device = device or cfg.get("embedding.device", "cpu")
        self.normalize = cfg.get("embedding.normalize_embeddings", True)
        self.dimension = 384
        self._model = None
        self._loaded = False
        self._lock = threading.Lock()

    def _load_model(self):
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    logger.info("Loading embedding model: %s on %s...", self.model_name, self.device)
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                    self.dimension = self._model.get_sentence_embedding_dimension() or 384
                    self._loaded = True
                    logger.info("Embedding model loaded successfully. Dimension: %d", self.dimension)

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of text passages. E5 model recommends 'passage: ' prefix for documents.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        is_e5 = "e5" in self.model_name.lower()
        formatted_texts = [f"passage: {t}" if is_e5 and not t.startswith("passage: ") else t for t in texts]
        self._load_model()
        
        embeddings = self._model.encode(
            formatted_texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a user search query string. E5 model recommends 'query: ' prefix for queries.
        """
        is_e5 = "e5" in self.model_name.lower()
        formatted_query = f"query: {query}" if is_e5 and not query.startswith("query: ") else query
        self._load_model()
        
        embedding = self._model.encode(
            [formatted_query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embedding.astype(np.float32)
