"""Real voice: what the offline recogniser heard in actual flights, run through
the shipped model and the plugin's decision rules (tests/fixtures/real_voice.jsonl,
built by scripts/build_real_voice_set.py from hand-labelled recordings).

The bar is the gate's (scripts/model_gate.py): never act on a misheard
command more often than v11 did, and get at least as many right.
Needs torch and a trained model, so it is skipped in CI.
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# v11 on this set: 70 right, 7 safely rejected, 1 wrong
# ("parking bring on engine" -> engine 1 start).
V11_RIGHT, V11_WRONG = 70, 1


@pytest.mark.requires_torch
def test_latest_model_on_real_voice():
    import model_gate
    from vimaan_nlu.model_loader import ModelLoader  # noqa: F401  (torch present)

    models = os.path.join(ROOT, "ML", "models", "vimaan_nlu_model_best")
    if not os.path.isdir(models):
        pytest.skip("no trained model checkout")
    latest = max((d for d in os.listdir(models) if d[1:].isdigit()), key=lambda d: int(d[1:]))
    result = model_gate.real_voice(model_gate.load(latest))
    assert result["wrong"] <= V11_WRONG, result["examples"]["wrong"]
    assert result["right"] >= V11_RIGHT
