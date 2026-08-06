"""
Unit tests verifying strict document and retrieval isolation between separate sessions.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import src.core.win_fix

import shutil
import numpy as np
from unittest.mock import patch, MagicMock

import sentence_transformers

mock_st = MagicMock()
mock_st.encode.return_value = np.zeros((1, 384), dtype=np.float32)
mock_st.get_sentence_embedding_dimension.return_value = 384

with patch.object(sentence_transformers, "SentenceTransformer", return_value=mock_st):
    from fastapi.testclient import TestClient
    from src.main import app

client = TestClient(app)


def test_session_isolation_and_creation():
    r1 = client.post("/api/sessions/new")
    assert r1.status_code == 200
    sess1_id = r1.json()["id"]

    r2 = client.post("/api/sessions/new")
    assert r2.status_code == 200
    sess2_id = r2.json()["id"]

    assert sess1_id != sess2_id, "Session IDs must be unique!"

    tmp_dir = Path("data/tmp_tests")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_a = tmp_dir / "file_a.txt"
    file_b = tmp_dir / "file_b.txt"

    file_a.write_text("Project Alpha has a total budget of 500 million VND.", encoding="utf-8")
    file_b.write_text("Project Beta has a total budget of 900 million VND.", encoding="utf-8")

    try:
        with open(file_a, "rb") as f:
            u1 = client.post(f"/api/sessions/{sess1_id}/upload", files={"file": ("file_a.txt", f, "text/plain")})
            assert u1.status_code == 200

        with open(file_b, "rb") as f:
            u2 = client.post(f"/api/sessions/{sess2_id}/upload", files={"file": ("file_b.txt", f, "text/plain")})
            assert u2.status_code == 200

        res1 = client.post(f"/api/sessions/{sess1_id}/ask", json={"query": "What is Project Alpha budget?"})
        assert res1.status_code == 200
        data1 = res1.json()

        for src in data1.get("sources", []):
            assert src["source_file"] == "file_a.txt", f"Session A cited illegal file from another session: {src['source_file']}"
            assert src["source_file"] != "file_b.txt"

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_session_isolation_and_creation()
    print(" test_session_isolation.py PASSED!")
