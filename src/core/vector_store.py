"""
VectorStore - Manages per-session FAISS indices, mirrored to a Hugging Face Hub dataset repo for durability.
"""
import pickle
import logging
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from src.core.hf_storage import HFDatasetStorage

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages a SINGLE global FAISS index with metadata filtering for all sessions."""

    def __init__(self, base_dir: str = "data/global_vector_store"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.faiss"
        self.metadata_path = self.base_dir / "chunks.pkl"
        
        self.index = None
        # Mapping from integer ID to chunk dictionary
        self.chunks: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        
        self.storage = HFDatasetStorage.get_instance()
        self.load_global_index()

    def load_global_index(self) -> bool:
        """Load the global FAISS index and metadata map from disk or HF."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            self.storage.download_file("global_vector_store/index.faiss", self.index_path)
            self.storage.download_file("global_vector_store/chunks.pkl", self.metadata_path)

        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, "rb") as f:
                    self.chunks = pickle.load(f)
                self.next_id = max(self.chunks.keys()) + 1 if self.chunks else 0
                logger.info("[VectorStore] Global FAISS index loaded with %d vectors.", self.index.ntotal)
                return True
            except Exception as e:
                logger.error("[VectorStore] Failed to load Global FAISS index: %s", e)
        
        # Initialize empty if not found
        self.chunks = {}
        self.next_id = 0
        return False

    def _save_global_index(self) -> None:
        """Save the global index to disk and upload to HF."""
        if self.index is None:
            return
            
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.chunks, f)
            
        self.storage.upload_file(self.index_path, "global_vector_store/index.faiss")
        self.storage.upload_file(self.metadata_path, "global_vector_store/chunks.pkl")

    def build_or_update_session_index(self, session_id: str, new_vectors: np.ndarray, new_chunks: List[Dict[str, Any]]) -> int:
        """Add vectors to the global FAISS index, tracking metadata for filtering."""
        dim = new_vectors.shape[1]
        
        if self.index is None:
            # Create an IDMap around FlatIP to support remove_ids later if needed
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        n_new = new_vectors.shape[0]
        ids = np.arange(self.next_id, self.next_id + n_new, dtype=np.int64)
        
        # Add to FAISS with explicit IDs
        self.index.add_with_ids(new_vectors, ids)
        
        # Add to Metadata Map
        for i, chunk in enumerate(new_chunks):
            # Ensure session_id is in metadata for filtering
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            chunk["metadata"]["session_id"] = session_id
            self.chunks[int(ids[i])] = chunk
            
        self.next_id += n_new
        self._save_global_index()
        
        logger.info("[VectorStore] Global FAISS index updated. Added %d chunks for session '%s'. Total: %d", n_new, session_id, self.index.ntotal)
        return self.get_session_chunk_count(session_id)

    def search_session(self, query_vector: np.ndarray, session_id: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Search the global FAISS index, then apply Metadata Post-Filtering by session_id."""
        if self.index is None or self.index.ntotal == 0:
            return []

        # Retrieve a large batch from global index to ensure we find enough session-specific chunks
        k_search = min(top_k * 50, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx not in self.chunks:
                continue
                
            chunk_info = self.chunks[idx]
            # Post-Filtering by Metadata
            if chunk_info.get("metadata", {}).get("session_id") == session_id:
                chunk_copy = chunk_info.copy()
                chunk_copy["score"] = float(score)
                results.append(chunk_copy)
                
            if len(results) >= top_k:
                break

        return results

    def get_session_summary_chunks(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Return uniformly sampled chunks for this specific session."""
        session_chunks = [c for c in self.chunks.values() if c.get("metadata", {}).get("session_id") == session_id]
        if not session_chunks:
            return []

        sorted_chunks = sorted(session_chunks, key=lambda c: c.get("metadata", {}).get("page", 999))
        n = len(sorted_chunks)

        if n <= limit:
            return sorted_chunks

        step = (n - 1) / float(limit - 1)
        sampled_indices = [int(round(i * step)) for i in range(limit)]
        seen_idx = set()
        result = []
        for idx in sampled_indices:
            idx = min(idx, n - 1)
            if idx not in seen_idx:
                seen_idx.add(idx)
                result.append(sorted_chunks[idx])

        return result

    def get_session_chunk_count(self, session_id: str) -> int:
        return sum(1 for c in self.chunks.values() if c.get("metadata", {}).get("session_id") == session_id)

    def delete_session_store(self, session_id: str) -> bool:
        """Delete all vectors belonging to session_id from the global index."""
        if self.index is None:
            return False
            
        ids_to_remove = [idx for idx, c in self.chunks.items() if c.get("metadata", {}).get("session_id") == session_id]
        if not ids_to_remove:
            return True
            
        # Remove from metadata
        for idx in ids_to_remove:
            del self.chunks[idx]
            
        # Remove from FAISS index (requires IndexIDMap)
        self.index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
        
        self._save_global_index()
        logger.info("[VectorStore] Removed %d vectors for session '%s' from Global FAISS.", len(ids_to_remove), session_id)
        
        # Cleanup legacy isolated directory if it exists
        legacy_dir = Path("data/sessions_vector_stores") / session_id
        if legacy_dir.exists():
            import shutil
            shutil.rmtree(legacy_dir, ignore_errors=True)
            
        return True
