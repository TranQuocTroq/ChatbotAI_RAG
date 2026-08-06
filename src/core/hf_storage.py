"""
HFDatasetStorage - Durable persistence backend backed by a Hugging Face Hub dataset repository.

Why this exists:
Hugging Face Spaces containers use ephemeral local disk. Any file written to disk
(session metadata, FAISS indices, BM25 indices, uploaded documents) is lost whenever
the Space restarts, redeploys, or goes to sleep. This module mirrors those files to a
Hugging Face Hub dataset repo so state survives container restarts, and acts as the
system's "database" instead of local-only JSON/pickle files.

Note on authentication:
Writing to any Hugging Face Hub repository (including a dataset repo) always requires
an access token with "write" scope - this is a Hub API requirement. This token is used
exclusively for Hub read/write operations and is unrelated to LLM inference
(LLM generation runs fully locally, see local_llm_engine.py).

Configuration (environment variables):
- HF_TOKEN: Hugging Face access token with write permission on HF_DATASET_REPO_ID.
- HF_DATASET_REPO_ID: target dataset repo id, e.g. "your-username/docbrain-storage".

If either variable is missing, this backend is disabled and the app gracefully falls back to
local-disk-only persistence (functional, but not durable on ephemeral hosts).
"""
import os
import shutil
import logging
import threading
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

logger = logging.getLogger(__name__)


class HFDatasetStorage:
    """Thin wrapper around huggingface_hub for syncing local files to a HF Hub dataset repo."""

    _instance: Optional["HFDatasetStorage"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.repo_id = os.getenv("HF_DATASET_REPO_ID", "trong333tn/session_chatbotai_rag").strip()
        self.token = os.getenv("HF_TOKEN", "").strip()
        self.enabled = bool(self.repo_id and self.token)
        self.api = HfApi(token=self.token) if self.enabled else None

        if self.enabled:
            self._ensure_repo_exists()
        else:
            logger.warning(
                "[HFDatasetStorage] Disabled: HF_DATASET_REPO_ID or HF_TOKEN not set. "
                "Falling back gracefully to local-disk-only persistence (not durable on ephemeral hosts)."
            )

    @classmethod
    def get_instance(cls) -> "HFDatasetStorage":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resets the singleton instance for testing graceful degradation."""
        with cls._lock:
            cls._instance = None

    def _ensure_repo_exists(self) -> None:
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            try:
                self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", private=True)
            except Exception as e:
                logger.error("[HFDatasetStorage] Failed to create dataset repo '%s': %s", self.repo_id, e)
                self.enabled = False
        except Exception as e:
            logger.error("[HFDatasetStorage] Failed to verify dataset repo '%s': %s", self.repo_id, e)
            self.enabled = False

    def upload_file(self, local_path: Path, repo_path: str) -> bool:
        """Upload a single local file to repo_path inside the dataset repo."""
        if not self.enabled or not Path(local_path).exists():
            return False
        try:
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            return True
        except Exception as e:
            logger.error("[HFDatasetStorage] upload_file failed for '%s': %s", repo_path, e)
            return False

    def upload_folder(self, local_dir: Path, repo_prefix: str) -> bool:
        """Upload an entire local directory to repo_prefix inside the dataset repo."""
        if not self.enabled or not Path(local_dir).exists():
            return False
        try:
            self.api.upload_folder(
                folder_path=str(local_dir),
                path_in_repo=repo_prefix,
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            return True
        except Exception as e:
            logger.error("[HFDatasetStorage] upload_folder failed for '%s': %s", repo_prefix, e)
            return False

    def download_file(self, repo_path: str, local_path: Path) -> bool:
        """Download a single file from the dataset repo into local_path. Returns False if absent."""
        if not self.enabled:
            return False
        try:
            cached_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=repo_path,
                repo_type="dataset",
                token=self.token,
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cached_path, local_path)
            return True
        except EntryNotFoundError:
            return False
        except Exception as e:
            logger.error("[HFDatasetStorage] download_file failed for '%s': %s", repo_path, e)
            return False

    def delete_path(self, repo_path: str) -> bool:
        """Delete a file or folder from the dataset repo."""
        if not self.enabled:
            return False
        try:
            self.api.delete_folder(path_in_repo=repo_path, repo_id=self.repo_id, repo_type="dataset")
            return True
        except Exception:
            try:
                self.api.delete_file(path_in_repo=repo_path, repo_id=self.repo_id, repo_type="dataset")
                return True
            except Exception as e:
                logger.error("[HFDatasetStorage] delete_path failed for '%s': %s", repo_path, e)
                return False
