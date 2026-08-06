"""
Retriever - Hybrid Search (Dense + Sparse BM25) + RRF + Cross-Encoder Re-ranking.

Modern Two-Stage Retrieval architecture:
- Stage 1: FAISS Dense Search + BM25 Sparse -> Top-20 candidates via RRF (high recall)
- Stage 2: Cross-Encoder (ms-marco-MiniLM-L-6-v2) Re-ranking -> Top-K final chunks (high precision)
"""
import logging
from typing import List, Dict, Any

from src.core.embedder import Embedder
from src.core.vector_store import VectorStore
from src.core.bm25_store import BM25Store
from src.core.config import get_config

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-Encoder Re-ranker using ms-marco-MiniLM-L-6-v2.
    Accurately scores document chunk relevance after first-stage retrieval.
    """
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        try:
            from sentence_transformers import CrossEncoder
            # Lightweight ~117MB multilingual model, calibrated for Vietnamese
            self.__class__._model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512)
            self._loaded = True
            logger.info("[Reranker] Cross-Encoder mmarco-mMiniLMv2-L12-H384-v1 loaded successfully.")
        except Exception as e:
            logger.warning("[Reranker] Failed to load Cross-Encoder: %s. Falling back to RRF ranking.", e)
            self._loaded = False

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """Re-ranks candidate chunks using Cross-Encoder. Fallback: returns original input on error."""
        if not candidate_chunks:
            return []

        self._load_model()

        if not self._loaded or self.__class__._model is None:
            return candidate_chunks[:top_k]

        try:
            pairs = [[query, chunk.get("text", "")] for chunk in candidate_chunks]
            scores = self.__class__._model.predict(pairs)

            import math
            for idx, score in enumerate(scores):
                # Sigmoid to convert logit to 0.0-1.0 probability
                sigmoid_score = 1 / (1 + math.exp(-float(score)))
                candidate_chunks[idx]["rerank_score"] = float(score)
                candidate_chunks[idx]["score"] = round(sigmoid_score, 4)

            sorted_chunks = sorted(candidate_chunks, key=lambda x: x.get("score", 0), reverse=True)
            top_chunks = sorted_chunks[:top_k]

            return top_chunks
        except Exception as e:
            logger.warning("[Reranker] Exception during re-rank: %s. Falling back to RRF top-k.", e)
            return candidate_chunks[:top_k]


class Retriever:
    """
    Orchestrates modern Two-Stage Retrieval:
    Stage 1: Hybrid Search (Dense E5-small + BM25) with RRF -> Top-20 candidates (High Recall)
    Stage 2: Cross-Encoder Re-ranking -> Final Top-K (High Precision)
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore, bm25_store: BM25Store = None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store or BM25Store()
        self.reranker = Reranker.get_instance()

    def retrieve(self, query: str, session_id: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Executes Two-Stage Retrieval:
        1. Hybrid Search (FAISS Dense + BM25 Sparse) with RRF -> Top-20 candidates
        2. Cross-Encoder Re-ranking -> Final top_k chunks
        """
        if not query.strip() or not session_id:
            return []

        # ── STAGE 1: Hybrid Search with RRF ───────────────────────────────────────
        first_stage_k = min(20, max(top_k * 5, 10))

        # Dense Vector Search (E5-small)
        query_vector = self.embedder.encode_query(query)
        dense_results = self.vector_store.search_session(query_vector, session_id=session_id, top_k=first_stage_k)

        # Sparse Keyword Search (BM25)
        sparse_results = self.bm25_store.search_session(query, session_id=session_id, top_k=first_stage_k)

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_k = 60.0

        for rank, chunk in enumerate(dense_results):
            chunk_id = f"{chunk.get('metadata', {}).get('source')}_p{chunk.get('metadata', {}).get('page')}_idx{chunk.get('metadata', {}).get('chunk_index')}"
            chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        for rank, chunk in enumerate(sparse_results):
            chunk_id = f"{chunk.get('metadata', {}).get('source')}_p{chunk.get('metadata', {}).get('page')}_idx{chunk.get('metadata', {}).get('chunk_index')}"
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        candidates = []
        for cid in sorted_ids[:first_stage_k]:
            chunk_data = chunk_map[cid].copy()
            raw_rrf = rrf_scores[cid]
            chunk_data["score"] = round(min(0.98, float(0.80 + raw_rrf * 5.0)), 4)
            candidates.append(chunk_data)

        if not candidates:
            return []

        # ── STAGE 2: Cross-Encoder Re-ranking ────────────────────────────────────
        final_chunks = self.reranker.rerank(query, candidates, top_k=top_k)
        return final_chunks
