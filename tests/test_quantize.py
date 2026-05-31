"""Tests for Phase 4 optimization code: inference max_length and the
quantization benchmark helpers.

All of these import modules that pull in torch at load time, so they are
gated on the ``requires_torch`` marker (auto-skipped in CI).
"""

import inspect

import pytest


@pytest.mark.requires_torch
class TestInferenceMaxLength:
    def test_default_is_32(self):
        from vimaan_nlu.inference import DEFAULT_MAX_LENGTH

        assert DEFAULT_MAX_LENGTH == 32

    def test_predict_accepts_max_length_kwarg(self):
        from vimaan_nlu.inference import DEFAULT_MAX_LENGTH, predict

        sig = inspect.signature(predict)
        assert "max_length" in sig.parameters
        assert sig.parameters["max_length"].default == DEFAULT_MAX_LENGTH


@pytest.mark.requires_torch
class TestPercentile:
    def test_p50_odd_length(self):
        from quantize_model import _percentile

        assert _percentile([1, 2, 3], 50) == 2

    def test_p0_and_p100(self):
        from quantize_model import _percentile

        vals = [5, 1, 3, 9, 7]
        assert _percentile(vals, 0) == 1
        assert _percentile(vals, 100) == 9

    def test_empty(self):
        from quantize_model import _percentile

        assert _percentile([], 95) == 0.0


@pytest.mark.requires_torch
class TestSlotPairF1:
    def test_perfect_match(self):
        from quantize_model import _slot_pair_f1

        true = [{"altitude": "10000"}, {"heading": "270"}]
        pred = [{"altitude": "10000"}, {"heading": "270"}]
        assert _slot_pair_f1(true, pred) == 1.0

    def test_disjoint(self):
        from quantize_model import _slot_pair_f1

        true = [{"altitude": "10000"}]
        pred = [{"heading": "270"}]
        assert _slot_pair_f1(true, pred) == 0.0

    def test_value_mismatch_counts_as_wrong(self):
        from quantize_model import _slot_pair_f1

        true = [{"frequency": "121.5"}]
        pred = [{"frequency": "121"}]
        assert _slot_pair_f1(true, pred) == 0.0

    def test_partial(self):
        from quantize_model import _slot_pair_f1

        # tp=1 (altitude), fp=1 (extra heading), fn=1 (missing flaps)
        true = [{"altitude": "10000", "flaps": "2"}]
        pred = [{"altitude": "10000", "heading": "270"}]
        # precision = 1/2, recall = 1/2 -> F1 = 0.5
        assert _slot_pair_f1(true, pred) == pytest.approx(0.5)

    def test_empty_slots_both(self):
        from quantize_model import _slot_pair_f1

        assert _slot_pair_f1([{}], [{}]) == 0.0
