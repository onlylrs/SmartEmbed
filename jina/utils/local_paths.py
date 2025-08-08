"""Utility for loading developer-specific local absolute paths.

Each developer creates a `local_paths.yaml` at project root (gitignored) with entries like:

model_base: /abs/path/to/jina-embeddings-v4-base
base_model_path: /abs/path/to/jina-embeddings-v4-base   # fallback key used elsewhere
raw_data: /abs/path/to/data/raw
processed_data: /abs/path/to/data/processed

Access with:
from jina.utils.local_paths import get_path
model_dir = get_path("model_base")

If the key is missing or file absent, returns None.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict
import yaml

_CACHE: Dict[str, str] | None = None


def _load_yaml() -> Dict[str, str]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    root = Path(__file__).resolve().parents[2]  # project root (jina/ is under root)
    candidate = root / "local_paths.yaml"
    if not candidate.exists():
        _CACHE = {}
        return _CACHE
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                data = {}
            _CACHE = {str(k): str(v) for k, v in data.items()}
    except Exception:
        _CACHE = {}
    return _CACHE


def get_path(key: str) -> Optional[str]:
    """Return path string for key from local_paths.yaml or environment variable override.

    Priority: ENV VAR (UPPERCASE) > YAML entry
    """
    env_key = key.upper()
    if env_key in os.environ:
        return os.environ[env_key]
    data = _load_yaml()
    return data.get(key)
