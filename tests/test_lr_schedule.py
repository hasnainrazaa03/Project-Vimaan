"""The trainer's linear learning-rate schedule (ML/utils/lr_schedule.py)."""

import pytest
from utils.lr_schedule import lr_factor


def test_warms_up_to_the_full_rate():
    assert lr_factor(0, 1000, 100) == pytest.approx(0.01)
    assert lr_factor(49, 1000, 100) == pytest.approx(0.5)
    assert lr_factor(99, 1000, 100) == pytest.approx(1.0)


def test_falls_linearly_to_zero_by_the_last_step():
    assert lr_factor(100, 1000, 100) == pytest.approx(1.0)
    assert lr_factor(550, 1000, 100) == pytest.approx(0.5)
    assert 0 < lr_factor(999, 1000, 100) < 0.002
    assert lr_factor(1000, 1000, 100) == 0.0
    assert lr_factor(5000, 1000, 100) == 0.0


def test_never_rises_after_warmup_and_stays_in_range():
    factors = [lr_factor(s, 1000, 60) for s in range(1000)]
    assert all(0 < f <= 1 for f in factors)
    peak = factors.index(max(factors))
    assert peak == 59
    assert all(a >= b for a, b in zip(factors[peak:], factors[peak + 1 :]))


def test_no_warmup_starts_at_the_full_rate():
    assert lr_factor(0, 10, 0) == pytest.approx(1.0)
    assert lr_factor(9, 10, 0) == pytest.approx(0.1)
