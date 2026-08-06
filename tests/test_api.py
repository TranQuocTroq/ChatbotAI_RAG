"""
Unit tests for FastAPI REST Endpoints in src/main.py.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import src.core.win_fix

import os
import numpy as np
from unittest.mock import patch, MagicMock

import sentence_transformers

# Mock SentenceTransformer encode to prevent slow model downloads during unit testing
mock_st = MagicMock()
mock_st.encode.return_value = np.zeros((1, 384), dtype=np.float32)
mock_st.get_sentence_embedding_dimension.return_value = 384

with patch.object(sentence_transformers, "SentenceTransformer", return_value=mock_st):
    from fastapi.testclient import TestClient
    from src.main import app

client = TestClient(app)


def test_get_sessions():
    """Test retrieving session metadata dictionary."""
    response = client.get("/api/sessions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_create_session():
    """Test creating a new session."""
    response = client.post("/api/sessions/new")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["id"].startswith("session_")
    assert data["doc_count"] == 0


def test_ask_chitchat_query():
    """Test asking a conversational chitchat query with mocked pipeline."""
    new_sess = client.post("/api/sessions/new").json()
    sess_id = new_sess["id"]

    mock_rag_result = {
        "answer": "Hello! Tôi là DocBrain AI.",
        "confidence": 1.0,
        "is_conversational": True,
        "sources": [],
        "execution_time_sec": 0.05
    }

    with patch("src.main.pipeline.ask", return_value=mock_rag_result):
        payload = {"query": "hello"}
        response = client.post(f"/api/sessions/{sess_id}/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer_short" in data
        assert "answer_full" in data
        assert data["confidence"] == 1.0


if __name__ == "__main__":
    test_get_sessions()
    test_create_session()
    test_ask_chitchat_query()
    print(" test_api.py PASSED!")
