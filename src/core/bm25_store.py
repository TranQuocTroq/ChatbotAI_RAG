"""
BM25Store - Manages per-session BM25 keyword indices, mirrored to a Hugging Face Hub dataset repo.
"""
import pickle
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
from src.core.hf_storage import HFDatasetStorage

logger = logging.getLogger(__name__)


from pyvi import ViTokenizer

def tokenize_text(text: str) -> List[str]:
    """Vietnamese Tokenizer using pyvi to preserve compound words (e.g., 'công_nghệ')."""
    if not text:
        return []
    clean_text = re.sub(r"[^\w\s]", " ", text.lower())
    tokenized = ViTokenizer.tokenize(clean_text)
    return [w for w in tokenized.split() if len(w) > 1]


class BM25Store:
    """Manages a per-session BM25 keyword index, mirrored to a Hugging Face Hub dataset repo."""

    def __init__(self, base_dir: str = "data/sessions_vector_stores"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.indices: Dict[str, Dict[str, Any]] = {}
        self.storage = HFDatasetStorage.get_instance()

    def _repo_path(self, session_id: str) -> str:
        return f"sessions_vector_stores/{session_id}/bm25.pkl"

    def build_or_update_session_bm25(self, session_id: str, chunks: List[Dict[str, Any]]) -> None:
        """Build and persist the BM25 index for session_id."""
        sess_dir = self.base_dir / session_id
        sess_dir.mkdir(parents=True, exist_ok=True)
        bm25_path = sess_dir / "bm25.pkl"

        corpus = [c["text"] for c in chunks]
        tokenized_corpus = [tokenize_text(doc) for doc in corpus]

        bm25_index = BM25Okapi(tokenized_corpus)

        with open(bm25_path, "wb") as f:
            pickle.dump({"index": bm25_index, "chunks": chunks}, f)

        self.indices[session_id] = {"index": bm25_index, "chunks": chunks}

        self.storage.upload_file(bm25_path, self._repo_path(session_id))
        logger.info("[BM25Store] Session '%s' BM25 index built and saved: %d documents.", session_id, len(chunks))

    def load_session_bm25(self, session_id: str) -> bool:
        """Loads session BM25 index from local disk or downloads from dataset storage."""
        sess_dir = self.base_dir / session_id
        bm25_path = sess_dir / "bm25.pkl"

        if not bm25_path.exists():
            self.storage.download_file(self._repo_path(session_id), bm25_path)

        if not bm25_path.exists():
            return False

        try:
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            self.indices[session_id] = data
            return True
        except Exception as e:
            logger.error("[BM25Store] Failed to load BM25 index for session '%s': %s", session_id, e)
            return False

    def search_session(self, query: str, session_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search exclusively within the BM25 index owned by session_id."""
        if session_id not in self.indices:
            loaded = self.load_session_bm25(session_id)
            if not loaded:
                return []

        bm25_data = self.indices[session_id]
        bm25_index = bm25_data["index"]
        chunks = bm25_data["chunks"]

        if not chunks:
            return []

        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []

        scores = bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_copy = chunks[idx].copy()
                chunk_copy["bm25_score"] = float(scores[idx])
                results.append(chunk_copy)

        return results
