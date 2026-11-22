"""Callouts (vimaan_nlu.callouts, Phase F.4): what is said when, rate-limited,
quiet on the ground and in the last 100 ft, and on the flights recorded in
X-Plane (tests/fixtures/flights) only where something was wrong."""

import json
from pathlib import Path

import pytest
from vimaan_nlu.callouts import REPEAT_SEC, SAID, Callouts, due
from vimaan_nlu.envelope import Limits, margins
from vimaan_nlu.flight_phase import PhaseTracker, sample_from_event

CLEAR = dict.fromkeys(("stall", "flaps", "gear", "vno", "vne", "mmo"))


def m(**kw):
    return {**CLEAR, **kw}


@pytest.mark.parametrize(
    ("margin", "key"),
    [
        (m(stall=3.0), "speed"),
        (m(vne=-1.0), "overspeed"),
        (m(mmo=-0.01), "mach"),
        (m(gear=-4.0), "gear_speed"),
        (m(flaps=4.0), "flap_speed"),  # within 5 kt: before the limit
        (m(flaps=-10.0), "flap_speed"),
    ],
)
def test_each_callout_fires_on_its_margin(margin, key):
    assert due("climb", margin, False, 3000.0, 0.0) == [key]
    assert SAID[key]


def test_inside_the_limits_nothing_is_said():
    ok = m(stall=40.0, flaps=20.0, gear=30.0, vno=-5.0, vne=60.0, mmo=0.1)  # Vno: smooth air
    assert due("cruise", ok, False, 5000.0, 0.0) == []


def test_the_gear_callout_is_on_an_approach_below_500_ft():
    assert due("approach", m(), False, 450.0, 0.0) == ["gear"]
    assert due("approach", m(), False, 450.0, 1.0) == []  # gear down
    assert due("approach", m(), False, 800.0, 0.0) == []  # above 500 ft
    assert due("go_around", m(), False, 450.0, 0.0) == []  # gear coming up: right


@pytest.mark.parametrize(
    ("phase", "on_ground", "agl_ft"),
    [
        ("takeoff_roll", True, 0.0),
        ("rollout", True, 0.0),
        ("taxi", True, 0.0),
        ("landing", False, 60.0),  # the flare is slow on purpose
        ("approach", False, 80.0),  # the last 100 ft whatever the phase says
        ("unknown", False, 3000.0),
        (None, False, 3000.0),
        ("climb", None, 3000.0),  # on the ground or not: unknown
    ],
)
def test_nothing_is_said_on_the_ground_or_in_the_last_hundred_feet(phase, on_ground, agl_ft):
    assert due(phase, m(stall=-5.0, vne=-5.0), on_ground, agl_ft, 0.0) == []


def test_the_most_urgent_first_and_each_once_every_30_s():
    c = Callouts()
    bad = m(stall=2.0, flaps=-3.0)
    assert c.update(0.0, "approach", bad, False, 900.0, 1.0) == "speed"
    assert c.update(1.0, "approach", bad, False, 900.0, 1.0) == "flap_speed"  # the next one
    assert c.update(2.0, "approach", bad, False, 900.0, 1.0) is None  # both said
    assert c.update(REPEAT_SEC, "approach", bad, False, 900.0, 1.0) == "speed"  # still slow
    c.reset()  # a new flight
    assert c.update(REPEAT_SEC + 1, "approach", bad, False, 900.0, 1.0) == "speed"


# ---- on the recorded flights -------------------------------------------------
FIXTURES = Path(__file__).parent / "fixtures" / "flights"
# The limits the plugin had for each (the session events, X-Plane 12.4.4).
LIMITS = {
    "be58": Limits(84, 73, 152, 122, 2, 152, 195, 223, 0.0, True),
    "be9l": Limits(100, 90, 190, 150, 2, 185, 169, 226, 0.46, True),
    "b738": Limits(139, 114, 250, 162, 8, 270, 280, 340, 0.82, True),
    "a333": Limits(175, 125, 215, 180, 4, 250, 330, 330, 0.86, True),
    "md82": Limits(150, 115, 280, 195, 5, 999, 340, 340, 0.95, True),  # 2026-11-22
}
# What would have been said: only where something was wrong.
SAID_ON = {
    # Slowing to its stall on a fixed throttle: "Speed" at 88 kt, 8 s before
    # its autopilot let go at 81 kt and it flew into the ground.
    "be58_approach_slowed_to_the_stall": {"speed"},
    # Left at takeoff thrust to 1,000 ft: through its flaps-1 limit (215 kt).
    "a333_takeoff": {"flap_speed"},
    # Full throttle after the go-around: over Vne levelling off.
    "be9l_go_around": {"overspeed"},
    # The air start 14 nm out sank to 222 ft over the hills, gear up, on the
    # approach: "Gear" (2026-11-22).
    "md82_balloon_on_the_glideslope": {"gear"},
}


def _callouts(name):
    """The callouts a recorded flight gives, replayed with its aircraft's
    limits (the fixtures have no Mach: no Mach callout)."""
    limits = LIMITS[name.split("_")[0]]
    tracker, c, said = PhaseTracker(), Callouts(), set()
    for raw in (FIXTURES / f"{name}.jsonl").read_text().splitlines():
        ev = json.loads(raw)
        s = sample_from_event(ev)
        tracker.update(s)
        mg = margins(limits, s.ias_kt, None, s.flaps, s.gear, s.on_ground)
        key = c.update(s.t, tracker.phase, mg, s.on_ground, s.agl_ft, s.gear)
        if key:
            said.add(key)
    return said


@pytest.mark.parametrize("name", sorted(p.stem for p in FIXTURES.glob("*.jsonl")))
def test_on_a_recorded_flight_only_what_was_wrong_is_called_out(name):
    assert _callouts(name) == SAID_ON.get(name, set())
