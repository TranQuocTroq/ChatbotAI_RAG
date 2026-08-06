"""
RAGPipeline - Orchestrates the 4-layer Agentic RAG flow:
- Layer 1: QueryRouter (LLM & vector-based agentic intent routing)
- Layer 2: System self-context & multi-turn chat history injection
- Layer 3: Selective retrieval (hybrid E5 dense + BM25 sparse + cross-encoder rerank)
- Layer 4: Generative LLM synthesis with Corrective RAG (CRAG)
"""
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from src.core.config import get_config
from src.core.document_processor import DocumentProcessor
from src.core.chunker import Chunker
from src.core.embedder import Embedder
from src.core.vector_store import VectorStore
from src.core.bm25_store import BM25Store
from src.core.retriever import Retriever
from src.core.llm_reader import LLMReader
from src.core.router import QueryRouter
from src.core.translator import CrossLingualTranslator
from src.core.cache import SemanticCache
import json
import re

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrator for the modern 4-layer Agentic RAG pipeline."""

    def __init__(self):
        self.cfg = get_config()
        self.processor = DocumentProcessor()
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()
        self.retriever = Retriever(self.embedder, self.vector_store, self.bm25_store)
        self.llm_reader = LLMReader()
        self.router = QueryRouter(embedder=self.embedder)
        self.translator = CrossLingualTranslator()
        self.semantic_cache = SemanticCache()
        self.answer_score_threshold = self.cfg.get("retrieval.score_threshold", 0.3)

    def process_and_ingest_file_for_session(self, session_id: str, file_path: str) -> Dict[str, Any]:
        """Ingests a single file into the specified session's vector & BM25 indices."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("[RAGPipeline] Session '%s' ingesting file (column-aware): %s", session_id, path.name)

        pages_data = self.processor.extract_text_with_pages(str(path))
        chunks = self.chunker.split_pages_into_chunks(pages_data, session_id=session_id)
        if not chunks:
            return {"session_id": session_id, "new_chunks": 0, "total_chunks": 0}

        texts = [c["text"] for c in chunks]
        vectors = self.embedder.encode_passages(texts)
        total_vectors = self.vector_store.build_or_update_session_index(session_id, vectors, chunks)

        all_session_chunks = [c for c in self.vector_store.chunks.values() if c.get("metadata", {}).get("session_id") == session_id]
        self.bm25_store.build_or_update_session_bm25(session_id, all_session_chunks)

        return {
            "session_id": session_id,
            "filename": path.name,
            "new_chunks": len(chunks),
            "total_chunks": total_vectors
        }

    def ask(self, query: str, session_id: str, session_doc_count: int = 0, session_docs: List[str] = None, top_k: int = 4, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Executes full multi-stage RAG query processing and answer generation."""
        start_time = time.time()

        if not query.strip():
            return {
                "query": query, "answer": "Please enter a question.",
                "session_id": session_id, "confidence": 0.0, "sources": [], "execution_time_sec": 0.0
            }

        docs = session_docs or []
        if not docs:
            session_raw_dir = Path("data/sessions_raw") / session_id
            if session_raw_dir.exists():
                docs = [f.name for f in session_raw_dir.glob("*") if f.is_file()]
        doc_cnt = session_doc_count or len(docs)

        # Continuous multi-signal document relevance score (top1_score & score_mean) from FAISS
        doc_summary_snippet = ""
        top1_score = 0.0
        score_mean = 0.0

        if doc_cnt > 0:
            sample_chunks = self.vector_store.get_session_summary_chunks(session_id, limit=10)
            if sample_chunks:
                doc_summary_snippet = " ".join([c.get("text", "")[:250] for c in sample_chunks])

            try:
                query_vec = self.embedder.encode_query(query)
                
                # LAYER 0: Semantic Cache Check
                cached_res = self.semantic_cache.get_cached_response(session_id, query_vec)
                if cached_res:
                    elapsed = round(time.time() - start_time, 2)
                    cached_res["execution_time_sec"] = elapsed
                    
                    # Structured Logging (Observability)
                    logger.info(json.dumps({
                        "event": "RAG_QUERY",
                        "session_id": session_id,
                        "latency_sec": elapsed,
                        "intent": "CACHE_HIT",
                        "sources_used": len(cached_res.get("sources", [])),
                        "cache_hit": True
                    }))
                    
                    return cached_res

                top_chunks = self.vector_store.search_session(query_vec, session_id, top_k=3)
                if top_chunks:
                    top1_score = float(top_chunks[0].get("score", 0.0))
                    scores = [float(c.get("score", 0.0)) for c in top_chunks]
                    score_mean = float(np.mean(scores))
            except Exception as e:
                logger.warning("[RAGPipeline] Router pre-check vector similarity error: %s", e)

        # Layer 1: multi-signal intent routing
        intent = self.router.classify_intent(
            query,
            session_docs=docs,
            doc_summary_snippet=doc_summary_snippet,
            top1_score=top1_score,
            score_mean=score_mean
        )
        logger.info("[RAGPipeline] Session '%s' intent classified: '%s' for query: '%s'", session_id, intent, query)

        # Out-of-scope: politely decline and suppress any citations
        if intent == "OUT_OF_SCOPE":
            elapsed = round(time.time() - start_time, 2)
            return {
                "query": query,
                "answer": f"Sorry, the uploaded documents in the session do not contain information to answer the question **'{query}'**.",
                "session_id": session_id,
                "confidence": 0.0,
                "used_api": False,
                "is_conversational": True,
                "sources": [],
                "execution_time_sec": elapsed
            }

        # Layers 2 & 4: chitchat response (no retrieval needed)
        if intent == "CHITCHAT" or doc_cnt == 0:
            llm_resp = self.llm_reader.generate_chitchat_response(query, session_docs=docs, chat_history=chat_history)
            elapsed = round(time.time() - start_time, 2)
            return {
                "query": query, "answer": llm_resp["answer"], "session_id": session_id,
                "confidence": llm_resp["confidence"], "used_api": llm_resp.get("used_api", False),
                "is_conversational": True, "sources": [], "execution_time_sec": elapsed
            }

        # Layers 3 & 4: whole-document summarization (uniform page sampling)
        if intent == "META_DOC":
            summary_chunks = self.vector_store.get_session_summary_chunks(session_id, limit=12)
            llm_resp = self.llm_reader.generate_summary_response(query, context_chunks=summary_chunks, session_docs=docs)
            elapsed = round(time.time() - start_time, 2)
            return {
                "query": query, "answer": llm_resp["answer"], "session_id": session_id,
                "confidence": llm_resp["confidence"], "used_api": llm_resp.get("used_api", False),
                "is_conversational": True, "sources": [], "execution_time_sec": elapsed
            }

        sample_chunks = self.vector_store.get_session_summary_chunks(session_id, limit=2)
        sample_text = sample_chunks[0].get("text", "") if sample_chunks else ""

        doc_lang = self.translator.detect_language(sample_text)
        query_lang = self.translator.detect_language(query)

        search_query = query
        if query_lang != doc_lang:
            # [Optimization Phase 4] Do not use LLM to translate query to save latency.
            # The multilingual-e5-small embedding can handle cross-lingual retrieval.
            pass

        # Retrieve top candidates
        context_chunks = self.retriever.retrieve(search_query, session_id=session_id, top_k=top_k)

        # Layer 4: generative RAG synthesis and Corrective RAG (CRAG)
        llm_resp = self.llm_reader.generate_rag_answer(
            query=query, context_chunks=context_chunks, session_docs=docs, chat_history=chat_history
        )

        # Extract citations
        sources = []
        seen_keys = set()
        top_score = float(context_chunks[0].get("score", 0.0)) if context_chunks else 0.0
        is_conv_or_not_found = (
            llm_resp.get("is_conversational", False) or
            llm_resp.get("confidence", 0.0) == 0.0 or
            top_score < self.answer_score_threshold
        )
        used_chunks = llm_resp.get("used_chunks", context_chunks)

        allowed_files = set(session_docs or [])

        if not is_conv_or_not_found:
            for idx, chunk in enumerate(used_chunks):
                meta = chunk.get("metadata", {})
                src_file = meta.get("source", "Unknown")

                if allowed_files and src_file not in allowed_files:
                    continue  # skip sources that don't belong to this session - never cite them

                page = meta.get("page", 1)
                key = (src_file, page)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # Layer 5: citation validation via n-gram overlap
                # Verify that the LLM answer actually contains keywords from the source chunk
                # Simple heuristic: remove punctuation and lower case to check word overlap
                ans_clean = re.sub(r'[^\w\s]', '', llm_resp["answer"].lower())
                chunk_clean = re.sub(r'[^\w\s]', '', chunk.get("text", "").lower())
                ans_words = set(ans_clean.split())
                chunk_words = set(chunk_clean.split())
                
                # If there is very little overlap (less than 3 meaningful words matching), the LLM likely hallucinated or ignored this chunk.
                # Skip words less than 4 chars (stopwords)
                ans_meaningful = {w for w in ans_words if len(w) > 3}
                chunk_meaningful = {w for w in chunk_words if len(w) > 3}
                
                overlap = ans_meaningful.intersection(chunk_meaningful)
                if len(overlap) < 2 and len(ans_meaningful) > 5:
                    logger.debug("[RAGPipeline] Dropped citation '%s' (pg %d) due to low verification overlap.", src_file, page)
                    continue

                sources.append({
                    "id": len(sources) + 1,
                    "source_file": src_file,
                    "page": page,
                    "relevance_score": round(float(chunk.get("score", 0.0)), 4),
                    "text_snippet": chunk.get("text", "")
                })

        elapsed = round(time.time() - start_time, 2)

        final_res = {
            "query": query,
            "answer": llm_resp["answer"],
            "session_id": session_id,
            "confidence": llm_resp["confidence"],
            "used_api": llm_resp.get("used_api", False),
            "is_conversational": is_conv_or_not_found,
            "sources": sources,
            "execution_time_sec": elapsed
        }

        # Cache the result if it was a successful Retrieval or MetaDoc intent
        if intent in ["RETRIEVAL", "META_DOC"] and not is_conv_or_not_found and doc_cnt > 0:
            if 'query_vec' in locals():
                self.semantic_cache.add_to_cache(session_id, query, query_vec, final_res)

        # Structured Logging (Observability)
        logger.info(json.dumps({
            "event": "RAG_QUERY",
            "session_id": session_id,
            "latency_sec": elapsed,
            "intent": intent,
            "sources_used": len(sources),
            "cache_hit": False,
            "confidence": llm_resp["confidence"]
        }))

        return final_res
