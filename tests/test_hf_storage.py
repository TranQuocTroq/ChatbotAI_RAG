"""
Unit tests for HFDatasetStorage graceful degradation and fallback behavior.
"""
import os
import sys
import logging
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).parent.parent))

from src.core.hf_storage import HFDatasetStorage


def test_graceful_degrade_when_env_vars_missing():
    """
    Verifies that HFDatasetStorage gracefully disables itself without crashing
    when HF_TOKEN or HF_DATASET_REPO_ID environment variables are empty or missing.
    """
    with patch.dict(os.environ, {"HF_TOKEN": "", "HF_DATASET_REPO_ID": ""}):
        HFDatasetStorage.reset_instance()
        storage = HFDatasetStorage.get_instance()
        
        assert storage.enabled is False
        assert storage.api is None
        
        # Operations should safely return False instead of raising exceptions
        assert storage.upload_file(Path("non_existent.txt"), "dummy/path") is False
        assert storage.upload_folder(Path("data"), "dummy/prefix") is False
        assert storage.download_file("dummy/path", Path("local.txt")) is False
        assert storage.delete_path("dummy/path") is False


def test_graceful_degrade_when_token_invalid():
    """
    Verifies that HFDatasetStorage handles invalid repository or API exceptions
    gracefully during initialization, disabling itself and logging an error.
    """
    with patch.dict(os.environ, {"HF_TOKEN": "invalid_token_xyz", "HF_DATASET_REPO_ID": "user/invalid_repo"}):
        HFDatasetStorage.reset_instance()
        with patch("huggingface_hub.HfApi.repo_info", side_effect=Exception("Unauthorized token")):
            storage = HFDatasetStorage.get_instance()
            
            assert storage.enabled is False
            assert storage.upload_file(Path("non_existent.txt"), "dummy/path") is False


if __name__ == "__main__":
    test_graceful_degrade_when_env_vars_missing()
    test_graceful_degrade_when_token_invalid()
    print(" test_hf_storage.py PASSED!")
