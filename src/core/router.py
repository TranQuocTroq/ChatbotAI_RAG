"""
QueryRouter - Enterprise Production Decision Engine (Multi-Signal & Out-Of-Scope Intent Routing).

Classifies user query into 4 distinct intent categories:
1. CHITCHAT: Conversational, greetings, polite talk -> LLM generation without retrieval.
2. OUT_OF_SCOPE: Specific lookup queries that lack relevant context in uploaded files -> Polite rejection without hallucination.
3. META_DOC: Whole-document summary / file listing requests -> Uniform sampling summarization pipeline.
4. RETRIEVAL: Specific facts contained within uploaded documents -> Two-stage hybrid retrieval with citation.
"""
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class QueryRouter:
    """Enterprise Production Decision Engine classifying query into 4 distinct intents."""

    def __init__(self, embedder=None):
        self.embedder = embedder
        self.centroids_initialized = False
        self.meta_doc_vecs = []
        self.chitchat_vecs = []
        self.retrieval_vecs = []

    def _init_centroids(self):
        """Initializes conversation & task vector clusters."""
        if self.centroids_initialized or not self.embedder:
            return

        meta_doc_samples = [
            "Summarize this entire document for me",
            "Summary report of the file's main content",
            "Show me the document overview",
            "Show me the list of files in the session",
            "Document này tổng quan nói về vấn đề gì vậy",
            "Briefly summarize the main points of the document"
        ]

        chitchat_samples = [
            "Hello",
            "Hello bạn",
            "Hi bot",
            "Hello",
            "Hi",
            "Hey",
            "Alo",
            "Thank you",
            "Thanks",
            "Ok thanks",
            "What is your name",
            "Who are you",
            "Today I feel tired and sad",
            "How is the weather today",
            "Nice to talk to you"
        ]

        retrieval_samples = [
            "What is the purchasing volume in tons",
            "How much percentage of cost does the model deployment save",
            "How much percentage does employee performance increase",
            "What mechanism does the system use to resolve hallucinations",
            "Which two search mechanisms does RAG technology combine",
            "What is the revenue and profit in billions",
            "What is the position of the general director",
            "What are the causes and technical solutions"
        ]

        try:
            self.meta_doc_vecs = [self.embedder.encode_query(s) for s in meta_doc_samples]
            self.chitchat_vecs = [self.embedder.encode_query(s) for s in chitchat_samples]
            self.retrieval_vecs = [self.embedder.encode_query(s) for s in retrieval_samples]
            self.centroids_initialized = True
        except Exception as e:
            logger.warning("[QueryRouter] Failed to initialize semantic centroids: %s", e)

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        a = np.array(vec_a).flatten()
        b = np.array(vec_b).flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _max_similarity(self, query_vec: np.ndarray, sample_vecs: List[np.ndarray]) -> float:
        if not sample_vecs:
            return 0.0
        return max(self._cosine_similarity(query_vec, v) for v in sample_vecs)

    def classify_intent(self, query: str, session_docs: List[str] = None, doc_summary_snippet: str = "", top1_score: float = 0.0, score_mean: float = 0.0) -> str:
        """
        Enterprise Decision Engine (Separates CHITCHAT vs OUT_OF_SCOPE vs META_DOC vs RETRIEVAL).
        """
        query_strip = query.strip()
        if not query_strip:
            return "CHITCHAT"

        has_docs = session_docs and len(session_docs) > 0

        # Fallback if no embedder
        if not self.embedder:
            return "RETRIEVAL" if top1_score >= 0.32 else "OUT_OF_SCOPE"

        if not self.centroids_initialized:
            self._init_centroids()

        # ── STEP 1: Semantic Intent Classifier & Confidence Score ────────────
        query_vec = self.embedder.encode_query(query_strip)

        score_meta = self._max_similarity(query_vec, self.meta_doc_vecs)
        score_chitchat = self._max_similarity(query_vec, self.chitchat_vecs)
        score_retrieval = self._max_similarity(query_vec, self.retrieval_vecs)

        scores = {
            "META_DOC": score_meta,
            "CHITCHAT": score_chitchat,
            "RETRIEVAL": score_retrieval
        }

        # ── STEP 2: Separate CHITCHAT vs META_DOC vs OUT_OF_SCOPE ────────────
        # 1. Prioritize CHITCHAT if score is high and confident (>= 0.70)
        if score_chitchat >= 0.70 and score_chitchat > score_meta and score_chitchat > score_retrieval:
            return "CHITCHAT"

        # 2. Prioritize META_DOC if user asks for whole-document summary (>= 0.70)
        if score_meta >= 0.70 and score_meta > score_chitchat and score_meta > score_retrieval:
            return "META_DOC"

        # ── STEP 3: Multi-Signal Document Relevance Check ─────────────────────
        # Combines top1_score and score_mean of top 3 chunks
        if has_docs:
            if top1_score >= 0.32 or (top1_score >= 0.28 and score_mean >= 0.25):
                return "RETRIEVAL"
            else:
                return "OUT_OF_SCOPE"

        return "CHITCHAT"
