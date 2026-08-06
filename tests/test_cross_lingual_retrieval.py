"""
Unit tests for CrossLingualTranslator and cross-lingual RAG retrieval.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import src.core.win_fix

import shutil
from src.core.translator import CrossLingualTranslator
from src.core.rag_pipeline import RAGPipeline


def test_language_detection_and_translation():
    translator = CrossLingualTranslator()
    
    assert translator.detect_language("What is the revenue in 2025?") == "vi"
    assert translator.detect_language("What is the total revenue in 2025?") == "en"
    
    translated_en = translator.translate_query("doanh thu năm 2025 là bao nhiêu", target_lang="en")
    assert isinstance(translated_en, str)
    assert len(translated_en) > 0


def test_cross_lingual_retrieval():
    pipe = RAGPipeline()
    session_id = "test_cross_lingual_sess"
    
    tmp_dir = Path("data/tmp_tests")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sample_en_file = tmp_dir / "sample_en.txt"
    sample_en_file.write_text("The total revenue for fiscal year 2025 reached 50 million USD.", encoding="utf-8")

    try:
        pipe.process_and_ingest_file_for_session(session_id, str(sample_en_file))

        res = pipe.ask("doanh thu năm 2025 là bao nhiêu", session_id=session_id, session_doc_count=1, session_docs=["sample_en.txt"])
        
        assert res is not None
        assert "answer" in res
    finally:
        pipe.vector_store.delete_session_store(session_id)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_language_detection_and_translation()
    test_cross_lingual_retrieval()
    print(" test_cross_lingual_retrieval.py PASSED!")
