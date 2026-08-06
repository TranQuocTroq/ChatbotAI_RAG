"""
SessionManager - Reads/writes session metadata (sessions.json) and keeps it durable
by mirroring it to a Hugging Face Hub dataset repo via HFDatasetStorage.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any

from src.core.hf_storage import HFDatasetStorage

logger = logging.getLogger(__name__)

SESSIONS_FILE = Path("data/sessions.json")
SESSIONS_RAW_DIR = Path("data/sessions_raw")
SESSIONS_VECTOR_DIR = Path("data/sessions_vector_stores")
SESSIONS_FILE_REPO_PATH = "sessions.json"


def load_sessions_from_disk() -> Dict[str, Any]:
    """Loads session metadata dictionary from local disk or downloads snapshot from dataset repo."""
    SESSIONS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    if not SESSIONS_FILE.exists():
        # Local disk is ephemeral - recover the latest snapshot from the HF dataset repo if present.
        HFDatasetStorage.get_instance().download_file(SESSIONS_FILE_REPO_PATH, SESSIONS_FILE)

    if SESSIONS_FILE.exists():
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data and isinstance(data, dict):
                    return data
        except Exception as e:
            logger.error("[SessionManager] Failed to read sessions.json: %s", e)

    return {}


def save_sessions_to_disk(sessions_data: Dict[str, Any]) -> None:
    """Saves session metadata to local disk and mirrors to HF dataset storage."""
    try:
        SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions_data, f, ensure_ascii=False, indent=2)

        HFDatasetStorage.get_instance().upload_file(SESSIONS_FILE, SESSIONS_FILE_REPO_PATH)
    except Exception as e:
        logger.error("[SessionManager] Failed to write sessions.json: %s", e)
