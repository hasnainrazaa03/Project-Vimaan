"""What Vimaan's model makes of a phrase, offline: the same model and
prediction the voice worker uses (latest model, int8, CPU). The runner
compares a generated voice's result with this, so "altimeters two niner
niner two" heard for "altimeter two niner niner two" still counts as
understood when it means the same thing."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "ML") not in sys.path:
    sys.path.insert(0, str(_REPO / "ML"))


@lru_cache(maxsize=1)
def _loader() -> Any:
    import torch
    from vimaan_nlu.model_loader import ModelLoader

    loader = ModelLoader(torch.device("cpu"))
    loader.load_all(quantize=True)
    return loader


@lru_cache(maxsize=512)
def meaning(text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """(intent, sorted slots) for ``text``."""
    from vimaan_nlu.inference import predict

    lo = _loader()
    p = predict(text, lo.model, lo.tokenizer, lo.device, lo.intent_map_rev, lo.slot_map_rev)
    return p["intent"], tuple(sorted((k, str(v)) for k, v in (p["slots"] or {}).items()))


def same(result: dict[str, Any], phrase: str) -> bool:
    """Did a say result understand ``phrase`` as the model would from the text?"""
    slots = tuple(sorted((k, str(v)) for k, v in (result.get("slots") or {}).items()))
    return (result.get("intent"), slots) == meaning(phrase)
