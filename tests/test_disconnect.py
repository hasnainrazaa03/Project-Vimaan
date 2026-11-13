"""Autopilot-off diagnosis (vimaan_nlu.disconnect), without X-Plane."""

import pytest
from vimaan_nlu import disconnect as d


def test_dropped_needs_on_then_off():
    on = {"servos_on": 1, "autopilot_on": 1}
    assert d.dropped(on, {"servos_on": 0, "autopilot_on": 0})
    assert d.dropped(on, {"servos_on": 0, "autopilot_on": 1})
    assert not d.dropped(None, {"servos_on": 0})
    assert not d.dropped({"servos_on": 0}, {"servos_on": 0})
    assert not d.dropped({"servos_on": None}, {"servos_on": 0})


def test_stick_peak_keeps_the_sign_and_only_the_lookback():
    s = d.StickTrace(keep_sec=6.0, every_sec=0.0)
    s.add(0.0, {"pitch": -0.9})  # too old for a 4 s lookback at t=5
    s.add(2.0, {"pitch": 0.3, "roll": -0.25})
    s.add(4.0, {"pitch": 0.1, "roll": 0.05, "yaw": None})
    assert s.peak(5.0, lookback=4.0) == {"pitch": 0.3, "roll": -0.25}
    s.add(20.0, {"pitch": 0.0})  # everything older than keep_sec is dropped
    assert s.peak(20.0, lookback=100.0) == {"pitch": 0.0}


def test_stick_sampling_is_throttled():
    s = d.StickTrace(every_sec=0.1)
    assert s.due(0.0)
    s.add(0.0, {"pitch": 0.0})
    assert not s.due(0.05) and s.due(0.1)


def test_recent_commands():
    c = d.CommandTrace(keep_sec=6.0)
    c.add(1.0, "sim/autopilot/servos_toggle", by_vimaan=True)
    c.add(3.5, "sim/autopilot/servos_fdir_off", by_vimaan=False)
    assert c.recent(5.0, lookback=2.0) == [
        {"command": "sim/autopilot/servos_fdir_off", "sec_before": 1.5, "by_vimaan": False}
    ]


def diag(**kw):
    args = {"stick_peak": {}, "commands": [], "has_power": 1, "fail_warn": 0}
    return d.diagnose(**{**args, **kw})


BUTTON = {"command": "sim/autopilot/servos_fdir_off", "sec_before": 0.5, "by_vimaan": False}
OURS = {"command": "sim/autopilot/servos_toggle", "sec_before": 0.5, "by_vimaan": True}


@pytest.mark.parametrize(
    ("kw", "cause"),
    [
        ({"reloading": True, "commands": [BUTTON]}, "reload"),
        ({"probe_running": True, "has_power": 0}, "probe"),
        ({"vimaan_turned_off": True, "stick_peak": {"pitch": 0.8}}, "vimaan"),
        ({"commands": [OURS]}, "vimaan"),
        ({"commands": [OURS, BUTTON]}, "command"),  # someone else's too
        ({"has_power": 0, "stick_peak": {"pitch": 0.8}}, "power"),
        ({"fail_warn": 1}, "failure"),
        ({"commands": [BUTTON], "stick_peak": {"pitch": 0.8}}, "command"),
        ({"stick_peak": {"roll": -0.35}}, "stick"),
        ({"stick_peak": {"pitch": 0.1}}, "unknown"),  # below STICK_MOVED
        ({"has_power": None, "fail_warn": None}, "unknown"),
    ],
)
def test_causes_in_order(kw, cause):
    assert diag(**kw)["cause"] == cause


def test_detail_names_everything_seen():
    got = diag(commands=[BUTTON], stick_peak={"pitch": 0.62, "roll": 0.05})
    assert got["detail"] == "command sim/autopilot/servos_fdir_off; stick pitch +0.62"
    assert got["say"] == "Autopilot disconnected by a switch or button."
    assert diag()["detail"] is None and diag()["say"] == "Autopilot disconnected."


def test_silent_causes():
    for cause in ("reload", "probe", "vimaan"):
        assert d.CALLOUTS[cause] is None
    assert set(d.CALLOUTS) == set(d.CAUSES)


def test_watched_commands_are_unique_and_namespaced():
    assert len(set(d.WATCHED_COMMANDS)) == len(d.WATCHED_COMMANDS)
    assert all(n.startswith(("sim/autopilot/", "laminar/B738/")) for n in d.WATCHED_COMMANDS)


def test_overspeed_comes_after_a_command_and_before_the_stick():
    fast = "335 kt, limit 330"
    assert diag(overspeed=fast)["cause"] == "overspeed"
    assert diag(overspeed=fast)["detail"] == "overspeed 335 kt, limit 330"
    assert diag(overspeed=fast)["say"] == "Autopilot disconnected. Overspeed."
    assert diag(overspeed=fast, commands=[BUTTON])["cause"] == "command"
    assert diag(overspeed=fast, stick_peak={"pitch": 0.8})["cause"] == "overspeed"
