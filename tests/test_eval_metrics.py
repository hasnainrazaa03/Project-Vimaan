"""Per-class validation metrics the trainer streams each epoch."""

import pytest
from utils.eval_metrics import per_intent_accuracy, per_slot_f1


def test_per_intent_accuracy():
    intent_map = {"None": 0, "gear": 1, "flaps": 2}
    out = per_intent_accuracy([0, 1, 1, 1, 2], [0, 1, 2, 1, 2], intent_map)
    assert out == {
        "None": {"acc": 1.0, "n": 1},
        "gear": {"acc": pytest.approx(2 / 3), "n": 3},
        "flaps": {"acc": 1.0, "n": 1},
    }


def test_per_slot_f1_merges_b_and_i_tags_and_leaves_out_o():
    slot_map = {"O": 0, "B-altitude": 1, "I-altitude": 2, "B-degrees": 3}
    true = [0, 1, 2, 3, 0]
    pred = [0, 1, 1, 0, 3]  # altitude right (B/I swap is the same slot), degrees missed once
    out = per_slot_f1(true, pred, slot_map)
    assert set(out) == {"altitude", "degrees"}
    assert out["altitude"] == {"f1": 1.0, "n": 2}
    assert out["degrees"]["f1"] == 0.0 and out["degrees"]["n"] == 1


def test_lengths_must_match():
    with pytest.raises(ValueError):
        per_intent_accuracy([0, 1], [0], {"a": 0, "b": 1})
