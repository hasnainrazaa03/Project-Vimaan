"""Tests for the evaluator's slot pair-F1 (PR-A / audit T0.2).

The evaluator module imports torch + sklearn, so this is torch-gated (skipped
in CI). The metric logic itself is pure.
"""

import os
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ML"))

from evaluation.evaluator import ModelEvaluator  # noqa: E402

_slot_pair_metrics = ModelEvaluator._slot_pair_metrics


class TestSlotPairMetrics:
    def test_perfect_match(self):
        m = _slot_pair_metrics([{"altitude": "15000"}], [{"altitude": "15000"}])
        assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)
        assert m["slot_f1"] == 1.0

    def test_wrong_value_is_fp_and_fn(self):
        # the exact 7005-vs-7500 regression the old evaluator could not see
        m = _slot_pair_metrics([{"altitude": "7005"}], [{"altitude": "7500"}])
        assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)
        assert m["slot_f1"] == 0.0

    def test_missing_predicted_slot_is_fn(self):
        m = _slot_pair_metrics([{}], [{"state": "up"}])
        assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 1)

    def test_extra_predicted_slot_is_fp(self):
        m = _slot_pair_metrics([{"state": "up", "degrees": "90"}], [{"state": "up"}])
        assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)

    def test_values_normalized_before_compare(self):
        # "5" and 5 compare equal after normalize_slot_value
        m = _slot_pair_metrics([{"degrees": "5"}], [{"degrees": 5}])
        assert m["tp"] == 1 and m["slot_f1"] == 1.0

    def test_empty_sets(self):
        m = _slot_pair_metrics([{}], [{}])
        assert m["slot_f1"] == 0.0  # nothing to score
