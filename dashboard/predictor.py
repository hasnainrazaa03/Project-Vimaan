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

    loader = _CACHE.get(model_path)
    if loader is None:
        # Load OUTSIDE the global lock so a slow (multi-second) weight load does
        # not serialize predicts for other, already-cached models. Double-check
        # after loading in case another thread cached the same path meanwhile.
        new_loader = ModelLoader()
        new_loader.load_all(model_path)
        with _CACHE_LOCK:
            loader = _CACHE.get(model_path)
            if loader is None:
                _CACHE[model_path] = new_loader
                loader = new_loader

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
