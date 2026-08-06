import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    In-memory / Disk-backed Semantic Cache for RAG responses.
    Prevents redundant LLM generation and retrieval for repeated/similar queries within a session.
    """
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "semantic_cache.pkl"
        
        # Structure: dict[session_id, list[dict(query, vector, result)]]
        self.memory_cache: Dict[str, List[Dict[str, Any]]] = self._load_cache()
        self.similarity_threshold = 0.95

    def _load_cache(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error("[SemanticCache] Failed to load cache: %s", e)
        return {}

    def _save_cache(self) -> None:
        def save_worker():
            try:
                # Create a shallow copy or copy of keys to avoid dict size changed during iteration errors
                # during pickling. We pickle the memory_cache dict.
                cache_copy = {k: v.copy() for k, v in self.memory_cache.items()}
                with open(self.cache_file, "wb") as f:
                    pickle.dump(cache_copy, f)
            except Exception as e:
                logger.error("[SemanticCache] Failed to save cache: %s", e)
                
        import threading
        t = threading.Thread(target=save_worker, daemon=True)
        t.start()

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        a = np.array(vec_a).flatten()
        b = np.array(vec_b).flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_cached_response(self, session_id: str, query_vec: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Check if a highly similar query exists in the session's cache.
        Returns the cached result dictionary if similarity > 0.95.
        """
        if session_id not in self.memory_cache:
            return None

        for entry in self.memory_cache[session_id]:
            sim = self._cosine_similarity(query_vec, entry["vector"])
            if sim >= self.similarity_threshold:
                logger.info("[SemanticCache] Cache HIT (similarity: %.3f) for session '%s'", sim, session_id)
                # Clone result to avoid mutable state issues
                result = entry["result"].copy()
                result["is_cached"] = True
                return result
                
        return None

    def add_to_cache(self, session_id: str, query_text: str, query_vec: np.ndarray, result: Dict[str, Any]) -> None:
        """
        Add a new query and its full response pipeline result to the cache.
        """
        if session_id not in self.memory_cache:
            self.memory_cache[session_id] = []
            
        # Optional limit: keep max 50 queries per session to prevent unbound memory growth
        if len(self.memory_cache[session_id]) >= 50:
            self.memory_cache[session_id].pop(0)

        # Remove execution time and specific trace metadata so they look clean when retrieved
        cache_result = result.copy()
        if "execution_time_sec" in cache_result:
            del cache_result["execution_time_sec"]

        self.memory_cache[session_id].append({
            "query": query_text,
            "vector": query_vec,
            "result": cache_result
        })
        
        self._save_cache()
