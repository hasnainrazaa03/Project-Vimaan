"""Lazy model loader + predict wrapper for the dashboard.

Caches one ModelLoader per checkpoint path so switching models doesn't
re-load weights the dashboard has seen before.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from .paths import ML_DIR

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _ensure_ml_on_path() -> None:
    p = str(ML_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def predict(model_path: str, text: str) -> dict[str, Any]:
    """Run inference on `text` using the checkpoint at `model_path`."""
    if not Path(model_path).is_dir():
        raise FileNotFoundError(model_path)
    _ensure_ml_on_path()

    from vimaan_nlu import predict as nlu_predict
    from vimaan_nlu.model_loader import ModelLoader

    with _CACHE_LOCK:
        loader = _CACHE.get(model_path)
        if loader is None:
            loader = ModelLoader()
            loader.load_all(model_path)
            _CACHE[model_path] = loader

    result = nlu_predict(
        text,
        loader.model,
        loader.tokenizer,
        loader.device,
        loader.intent_map_rev,
        loader.slot_map_rev,
    )
    return {
        "intent": result["intent"],
        "confidence": float(result["confidence"]),
        "slots": result["slots"],
        "model": model_path,
    }


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
