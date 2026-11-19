"""Phase of flight on real flights (Phase F.3): samples recorded in X-Plane
12.4 by the plugin, once a second, replayed through today's tracker.

The flights are simtest's (``python -m simtest run takeoff|landing|go-around
<aircraft>``) and the probes flown while building them, on 2026-11-19, at
Portland (KPDX 10R), plus a Tier 1 air start west of Portland. Each fixture
is one flight from a load on: the tracker's fields only, the first 5 samples
dropped (the plugin skips them after a load: the airspeed reads 0 for a
couple of seconds), cut 20 samples after the last change.

A change to the tracker that changes one of these sequences changes what it
says about a real flight: look at the flight before updating the sequence.
Never tune the tracker to the go-around/landing details of one fixture alone.
"""

import json
from pathlib import Path

import pytest
from vimaan_nlu.flight_phase import replay, sample_from_event

FIXTURES = Path(__file__).parent / "fixtures" / "flights"

PHASES = {
    # A Tier 1 air start, then the checklist's 1,500 fpm descent and level
    # changes. At 2,400 ft above the hills, a dip in altitude hold (-574 fpm)
    # is a descent, not an approach.
    "be58_air_start_over_hills": [
        "climb",
        "descent",
        "cruise",
        "descent",
        "cruise",
        "descent",
        "cruise",
        "descent",
    ],
    # Takeoffs: the first 40 kt are a taxi.
    "be58_takeoff": ["parked", "taxi", "takeoff_roll", "initial_climb", "climb", "cruise"],
    "b738_takeoff": ["parked", "taxi", "takeoff_roll", "initial_climb", "climb", "cruise"],
    "a333_takeoff": ["parked", "taxi", "takeoff_roll", "initial_climb", "climb", "cruise"],
    # Landings from an 8 nm final. X-Plane starts it at 1,500 ft and -1,600
    # fpm; the autopilot's level-off is a climb (above 1,000 ft, never a
    # go-around).
    "be58_landing": [
        "descent",
        "approach",
        "climb",
        "cruise",
        "approach",
        "landing",
        "rollout",
        "taxi",
        "parked",
    ],
    # Touched down, bounced to 38 ft and floated: one landing, one rollout.
    "be9l_landing_bounce": [
        "descent",
        "approach",
        "climb",
        "cruise",
        "descent",
        "approach",
        "landing",
        "rollout",
        "taxi",
        "parked",
    ],
    # Braked from 40 kt to a stop inside a phase's 3 s: no taxi.
    "b738_landing": ["approach", "climb", "approach", "landing", "rollout", "parked"],
    # Autothrottle left on: it held 145 kt along the runway at 2 ft (still
    # the rollout below 50 ft), then climbed away (a touch-and-go), and
    # crashed off the end of the runway.
    "b738_landing_floats_and_climbs": [
        "approach",
        "climb",
        "approach",
        "landing",
        "rollout",
        "initial_climb",
        "parked",
    ],
    # Go-arounds at 500 ft: full thrust and a 1,500 fpm climb to 3,000 ft.
    "be58_go_around": [
        "descent",
        "approach",
        "climb",
        "cruise",
        "approach",
        "go_around",
        "climb",
        "cruise",
    ],
    "be9l_go_around": [
        "descent",
        "approach",
        "climb",
        "cruise",
        "descent",
        "approach",
        "go_around",
        "climb",
        "cruise",
    ],
    "b738_go_around": ["approach", "climb", "approach", "go_around", "climb", "cruise"],
    # The pitch "held" through the slow bridge: back into the runway from
    # 75 ft. An impact reads as the ground phases.
    "be58_takeoff_flown_into_the_runway": [
        "parked",
        "taxi",
        "takeoff_roll",
        "initial_climb",
        "taxi",
        "parked",
    ],
}


def _flight(name):
    rows = (FIXTURES / f"{name}.jsonl").read_text().splitlines()
    return [sample_from_event(json.loads(r)) for r in rows]


def test_every_fixture_has_its_phases():
    assert sorted(p.stem for p in FIXTURES.glob("*.jsonl")) == sorted(PHASES)


@pytest.mark.parametrize("name", sorted(PHASES))
def test_a_recorded_flight_replays_to_its_phases(name):
    assert [p for _, p in replay(_flight(name))] == PHASES[name]


def test_real_landings_have_no_go_around():
    for name in PHASES:
        if "landing" in name:
            assert "go_around" not in [p for _, p in replay(_flight(name))], name


def test_a_go_around_starts_low_and_ends_at_a_thousand_feet():
    for name in ("be58_go_around", "be9l_go_around", "b738_go_around"):
        samples = _flight(name)
        changes = replay(samples)
        start = next(t for t, p in changes if p == "go_around")
        end = next(t for t, p in changes if t > start)
        at = {s.t: s for s in samples}
        first = min((t for t in at if t >= start), default=None)
        last = max(t for t in at if t <= end)
        assert at[first].agl_ft is not None and at[first].agl_ft < 600, name  # from 500 ft
        assert at[last].agl_ft is not None and at[last].agl_ft >= 900, name  # to ~1,000 ft


def test_the_landing_phase_is_the_last_hundred_feet():
    samples = _flight("be58_landing")
    changes = dict((p, t) for t, p in replay(samples))
    at = next(s for s in samples if s.t >= changes["landing"])
    assert at.agl_ft is not None and at.agl_ft < 100
    assert not at.on_ground
