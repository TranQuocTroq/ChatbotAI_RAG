"""
Unit tests verifying file upload size limits, extension validation, and API key authentication.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import src.core.win_fix

import os
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


def test_file_upload_extension_validation():
    """Verifies that uploading unsupported file extensions (e.g. .exe) is rejected with HTTP 400."""
    sess = client.post("/api/sessions/new").json()
    sess_id = sess["id"]

    bad_file = ("script.exe", b"malicious content", "application/octet-stream")
    response = client.post(f"/api/sessions/{sess_id}/upload", files={"file": bad_file})
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_file_upload_size_limit():
    """Verifies that uploading files exceeding 100MB is rejected with HTTP 413 Payload Too Large."""
    sess = client.post("/api/sessions/new").json()
    sess_id = sess["id"]

    large_content = b"x" * (101 * 1024 * 1024)
    large_file = ("large_doc.pdf", large_content, "application/pdf")
    response = client.post(f"/api/sessions/{sess_id}/upload", files={"file": large_file})
    assert response.status_code == 413
    assert "exceeds maximum limit of 100MB" in response.json()["detail"]


def test_admin_api_key_authentication():
    """Verifies that destructive endpoints enforce X-API-Key when ADMIN_API_KEY env is set."""
    sess = client.post("/api/sessions/new").json()
    sess_id = sess["id"]

    with patch.dict(os.environ, {"ADMIN_API_KEY": "super_secret_admin_key_123"}):
        res_no_auth = client.delete(f"/api/sessions/{sess_id}")
        assert res_no_auth.status_code == 401
        assert "Unauthorized" in res_no_auth.json()["detail"]

        res_bad_auth = client.delete(f"/api/sessions/{sess_id}", headers={"X-API-Key": "wrong_key"})
        assert res_bad_auth.status_code == 401

        res_valid_auth = client.delete(f"/api/sessions/{sess_id}", headers={"X-API-Key": "super_secret_admin_key_123"})
        assert res_valid_auth.status_code == 200
        assert res_valid_auth.json()["status"] == "deleted"


if __name__ == "__main__":
    test_file_upload_extension_validation()
    test_file_upload_size_limit()
    test_admin_api_key_authentication()
    print(" test_file_validation_and_auth.py PASSED!")
