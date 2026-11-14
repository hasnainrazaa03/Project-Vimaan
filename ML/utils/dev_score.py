"""Dev scored the way the gate scores: what the plugin would do.

Dev loss is a poor guide once dev intent accuracy is ~99.8%: a handful of
confidently wrong rows decide it, and two or three of them make it spike
(v26-g: 0.019 -> 0.22 between two epochs). This counts outcomes instead, with
scripts/model_gate.py's own ``actions`` and ``negatives`` (the plugin's
checks), on the dev templates: never trained on, and not the gate's test.

score = right - wrong_weight * (wrong + negatives acted on). A wrong action is
counted as ``wrong_weight`` lost commands: a refusal costs the pilot a repeat,
a wrong action does something they didn't ask for.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

WRONG_WEIGHT = 5.0
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _gate():
    if str(_SCRIPTS) not in sys.path:
        sys.path.append(str(_SCRIPTS))
    import model_gate

    return model_gate


def score(run, rows: list[dict[str, Any]], wrong_weight: float = WRONG_WEIGHT) -> dict[str, Any]:
    """``run(text) -> prediction`` scored on ``rows`` (dataset rows)."""
    mg = _gate()
    act = mg.actions(run, rows)
    neg = mg.negatives(run, rows)
    wrong = act["wrong"] + neg["acted_count"]
    return {
        "n": act["n"],
        "right": act["right"],
        "rejected": act["rejected"],
        "wrong": act["wrong"],
        "negatives": neg["n"],
        "negatives_acted": neg["acted_count"],
        "score": act["right"] - wrong_weight * wrong,
    }


def live_runner(model, tokenizer, device, intent_map: dict, slot_map: dict):
    """A ``run`` for a model being trained (the plugin's own predict)."""
    from vimaan_nlu.inference import predict

    intent_rev = {v: k for k, v in intent_map.items()}
    slot_rev = {v: k for k, v in slot_map.items()}

    def run(text: str) -> dict[str, Any]:
        return predict(text, model, tokenizer, device, intent_rev, slot_rev)

    return run


def saved_runner(path: str | Path):
    """A ``run`` for a saved model, loaded as the plugin loads it."""
    return _gate().load(str(path))
