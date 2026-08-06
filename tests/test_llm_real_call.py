"""
Unit tests verifying local LLM engine generation and LLMReader RAG synthesis.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(str(Path(__file__).parent.parent))

from src.core.llm_reader import LLMReader


def test_llm_reader_rag_answer_synthesis():
    """Verifies LLMReader generates proper structured RAG answers with confidence scores."""
    reader = LLMReader()
    
    context_chunks = [
        {
            "text": "Vinamilk revenue in 2025 reached 60,000 billion VND.",
            "score": 0.85,
            "metadata": {"source": "report.pdf", "page": 1}
        }
    ]

    res = reader.generate_rag_answer(query="Doanh thu Vinamilk năm 2025 là bao nhiêu?", context_chunks=context_chunks)
    assert res is not None
    assert "answer" in res
    assert "confidence" in res
    assert res["confidence"] > 0.0


def test_llm_reader_out_of_doc_rejection():
    """Verifies LLMReader triggers Corrective RAG rejection when context score is low."""
    reader = LLMReader()
    
    low_score_chunks = [
        {
            "text": "Irrelevant information.",
            "score": 0.15,
            "metadata": {"source": "report.pdf", "page": 1}
        }
    ]

    res = reader.generate_rag_answer(query="Price of Airbus A350?", context_chunks=low_score_chunks)
    assert res is not None
    assert res["confidence"] == 0.0
    assert res["is_conversational"] is True
    assert "does not contain information" in res["answer"]


if __name__ == "__main__":
    test_llm_reader_rag_answer_synthesis()
    test_llm_reader_out_of_doc_rejection()
    print(" test_llm_real_call.py PASSED!")
