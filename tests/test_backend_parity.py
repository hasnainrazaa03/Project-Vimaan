"""Comparison rules of scripts/backend_parity.py, with fake backends (no torch)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import backend_parity  # noqa: E402

HDG = "set_autopilot_heading"


def backend(table):
    """A fake backend: text -> (intent, slots, confidence)."""

    def run(text):
        intent, slots, conf = table.get(text, ("None", {}, 0.99))
        return {"intent": intent, "slots": dict(slots), "confidence": conf, "normalized_text": text}

    return run


TEXTS = {
    "heading 270": (HDG, {"degrees": "270"}),
    "heading 090": (HDG, {"degrees": "090"}),
    "seventy": ("None", {}),
}
SAME = {
    "heading 270": (HDG, {"degrees": "270"}, 0.99),
    "heading 090": (HDG, {"degrees": "090"}, 0.99),
}


def test_identical_backends_never_differ():
    result = backend_parity.compare(TEXTS, {"a": backend(SAME), "b": backend(SAME)})
    assert result["n_texts"] == 3
    assert result["pairs"]["a_vs_b"]["differ"] == 0


def test_differences_are_scored_against_the_label():
    worse = {
        **SAME,
        "heading 270": (HDG, {"degrees": "270"}, 0.40),  # under the confidence floor
        "heading 090": (HDG, {"degrees": "180"}, 0.99),  # wrong value
    }
    pair = backend_parity.compare(TEXTS, {"a": backend(SAME), "b": backend(worse)})["pairs"][
        "a_vs_b"
    ]
    assert pair["differ"] == 2
    assert pair["scored"]["a"] == {"right": 2, "rejected": 0, "wrong": 0}
    assert pair["scored"]["b"] == {"right": 0, "rejected": 1, "wrong": 1}


def test_confidence_alone_is_not_a_difference():
    # Same decision with different confidence: the plugin does the same thing.
    shaky = {**SAME, "heading 270": (HDG, {"degrees": "270"}, 0.80)}
    result = backend_parity.compare(TEXTS, {"a": backend(SAME), "b": backend(shaky)})
    assert result["pairs"]["a_vs_b"]["differ"] == 0


def test_negative_is_right_when_rejected_and_wrong_when_acted_on():
    acted = backend({"heading seventy": (HDG, {"degrees": "070"}, 0.99)})("heading seventy")
    assert backend_parity.score(acted, ("None", {})) == "wrong"
    rejected = backend({})("heading seventy")  # predicted None: not acted on
    assert backend_parity.score(rejected, ("None", {})) == "right"
