"""Scoring rules of scripts/model_gate.py, with a fake model (no torch)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import model_gate  # noqa: E402

COM = "set_com_frequency"


def fake(slots):
    return lambda text: {
        "intent": COM,
        "slots": dict(slots),
        "confidence": 0.99,
        "normalized_text": text,
    }


def f1(run, row):
    return model_gate.score_rows(run, [row])[COM]["slot_f1"]


def test_no_port_spoken_scores_as_com_1():
    row = {"intent": COM, "text": "frequency 121.5", "slots": {"frequency": "121.5"}}
    assert f1(fake({"frequency": "121.5"}), row) == 1.0
    want = model_gate.canonical_expected(row)[1]
    assert model_gate.judge(fake({"frequency": "121.5"}), row["text"], COM, want)[0] == "right"


def test_explicit_com_two_tuned_as_com_1_stays_wrong():
    row = {
        "intent": COM,
        "text": "frequency 133.1 on com 2",
        "slots": {"frequency": "133.1", "com_port": "2"},
    }
    want = model_gate.canonical_expected(row)[1]
    assert want["com_port"] == "2"
    missed = fake({"frequency": "133.1"})  # the model didn't tag the port
    assert f1(missed, row) < 1.0
    assert model_gate.judge(missed, row["text"], COM, want)[0] == "wrong"
    tagged = fake({"frequency": "133.1", "com_port": "2"})
    assert f1(tagged, row) == 1.0
    assert model_gate.judge(tagged, row["text"], COM, want)[0] == "right"
