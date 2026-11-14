"""Train until dev stops improving, then settle (ML/utils/convergence.py)."""

import pytest
from utils.convergence import Convergence


def make(**kw):
    args = dict(patience=2, total_steps=1000, warmup_steps=50, decay_steps=100)
    args.update(kw)
    return Convergence(**args)


def test_warms_up_then_holds_the_full_rate():
    c = make()
    assert c.factor(0) == pytest.approx(1 / 50)
    assert c.factor(49) == pytest.approx(1.0)
    assert c.factor(50) == c.factor(500) == 1.0


def test_decays_once_dev_stops_improving_and_then_stops():
    c = make()
    assert c.check(100, 10.0)  # first check: best so far
    assert c.check(200, 12.0)
    assert not c.check(300, 11.0)  # no gain: 1 of 2
    assert not c.decaying
    assert not c.check(400, 12.0)  # level with the best: still no gain
    assert c.decaying and c.reason == "converged" and c.decay_start == 400
    assert c.factor(400) == 1.0
    assert c.factor(450) == pytest.approx(0.5)
    assert c.factor(500) == 0.0
    assert not c.finished(499) and c.finished(500)


def test_a_gain_resets_the_count():
    c = make()
    c.check(100, 10.0)
    c.check(200, 9.0)
    c.check(300, 11.0)  # a gain
    c.check(400, 10.5)
    assert not c.decaying and c.checks_without_gain == 1


def test_a_gain_smaller_than_min_gain_is_not_progress_but_is_still_saved():
    c = make(min_gain=3.0)
    c.check(100, 100.0)
    assert c.check(200, 102.0)  # the best weights so far: saved
    assert c.checks_without_gain == 1  # but not enough to count as improving
    c.check(300, 101.0)
    assert c.decaying


def test_a_run_that_keeps_improving_decays_at_the_cap():
    c = make()
    for step in range(1, 1000):
        c.step_done(step)
        if step % 100 == 0:
            c.check(step, float(step))
        if c.decaying:
            break
    assert c.reason == "cap" and c.decay_start == 900
    assert c.finished(1000)


def test_decay_must_fit_in_the_run():
    with pytest.raises(ValueError):
        make(decay_steps=0)
    with pytest.raises(ValueError):
        make(decay_steps=2000)


def test_dev_is_still_checked_while_decaying():
    c = make()
    c.check(100, 5.0)
    c.check(200, 4.0)
    c.check(300, 4.0)
    assert c.decaying
    assert c.check(350, 6.0)  # settling helped: these are the weights to keep
    assert c.decay_start == 300  # the decay isn't restarted
