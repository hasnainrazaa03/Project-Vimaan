"""Phase of flight (vimaan_nlu.flight_phase) on synthetic flights, one sample
a second, as the plugin takes them."""

import pytest
from vimaan_nlu.flight_phase import (
    DWELL_SEC,
    FlightSample,
    PhaseTracker,
    candidate,
    replay,
    sample_from_event,
)


class Flight:
    """Builds samples second by second: ``hold`` repeats the aircraft as it
    is; ``ramp`` moves its height and speed linearly."""

    def __init__(self, **state):
        self.t = 0.0
        self.state = {
            "on_ground": True,
            "agl_ft": 0.0,
            "gs_kt": 0.0,
            "ias_kt": 0.0,
            "vs_fpm": 0.0,
            "gear": 1.0,
            "flaps": 0.0,
        }
        self.state.update(state)
        self.samples: list[FlightSample] = []

    def hold(self, sec, **change):
        self.state.update(change)
        for _ in range(int(sec)):
            self.samples.append(FlightSample(t=self.t, **self.state))
            self.t += 1.0
        return self

    def ramp(self, sec, agl_to=None, gs_to=None, **change):
        """Height and ground speed move linearly over ``sec``; vertical speed
        follows the height."""
        self.state.update(change)
        agl0, gs0 = self.state["agl_ft"], self.state["gs_kt"]
        agl1 = agl0 if agl_to is None else agl_to
        gs1 = gs0 if gs_to is None else gs_to
        n = int(sec)
        for i in range(1, n + 1):
            self.state["agl_ft"] = agl0 + (agl1 - agl0) * i / n
            self.state["gs_kt"] = gs0 + (gs1 - gs0) * i / n
            self.state["ias_kt"] = self.state["gs_kt"]
            self.state["vs_fpm"] = (agl1 - agl0) / n * 60.0
            self.samples.append(FlightSample(t=self.t, **self.state))
            self.t += 1.0
        self.state["vs_fpm"] = 0.0
        return self

    def phases(self):
        return [phase for _, phase in replay(self.samples)]


def test_a_whole_flight_goes_through_every_phase_in_order():
    f = Flight()
    f.hold(20)  # at the gate
    f.ramp(30, gs_to=15).hold(60)  # taxi out
    f.ramp(20, gs_to=90)  # takeoff roll
    f.ramp(60, agl_to=1500, on_ground=False, gear=0.0)  # liftoff, initial climb
    f.ramp(300, agl_to=9000)  # climb
    f.hold(300)  # cruise
    f.ramp(240, agl_to=3000)  # descent
    f.ramp(120, agl_to=100, flaps=0.5, gear=1.0)  # approach
    f.ramp(10, agl_to=0, gs_to=70)  # landing (flare)
    f.hold(1, on_ground=True, agl_ft=0.0, vs_fpm=0.0)
    f.ramp(25, gs_to=15)  # rollout
    f.hold(30)  # taxi in
    f.ramp(10, gs_to=0).hold(20)  # parked
    assert f.phases() == [
        "parked",
        "taxi",
        "takeoff_roll",
        "initial_climb",
        "climb",
        "cruise",
        "descent",
        "approach",
        "landing",
        "rollout",
        "taxi",
        "parked",
    ]


def test_a_go_around_from_the_approach_climbs_away():
    f = Flight(on_ground=False, agl_ft=2000.0, gs_kt=120.0, flaps=0.5)
    f.ramp(60, agl_to=300)  # approach
    f.ramp(90, agl_to=1500)  # go around, then climb above 1,000 ft
    assert f.phases() == ["approach", "go_around", "climb"]


def test_a_touch_and_go_lifts_off_into_an_initial_climb():
    f = Flight(on_ground=False, agl_ft=1500.0, gs_kt=90.0, flaps=0.5)
    f.ramp(60, agl_to=50)
    f.ramp(5, agl_to=0)
    f.hold(1, on_ground=True, agl_ft=0.0)
    f.hold(10)  # rolling at 90 kt
    f.ramp(30, agl_to=600, on_ground=False)
    assert f.phases() == ["approach", "landing", "rollout", "initial_climb"]


def test_a_bounce_on_landing_is_not_a_takeoff():
    f = Flight(on_ground=False, agl_ft=1200.0, gs_kt=80.0, flaps=0.5)
    f.ramp(50, agl_to=60)
    f.ramp(5, agl_to=0)
    f.hold(1, on_ground=True, agl_ft=0.0)
    f.hold(1, on_ground=False, agl_ft=4.0)  # airborne again for under a second
    f.hold(20, on_ground=True, agl_ft=0.0)
    assert f.phases() == ["approach", "landing", "rollout"]


def test_a_long_bounce_below_fifty_feet_is_still_the_rollout():
    f = Flight(on_ground=False, agl_ft=1200.0, gs_kt=80.0, flaps=0.5)
    f.ramp(50, agl_to=60)
    f.ramp(5, agl_to=0)
    f.hold(2, on_ground=True, agl_ft=0.0)
    f.hold(3, on_ground=False, agl_ft=15.0, vs_fpm=200.0)  # three seconds back in the air
    f.hold(20, on_ground=True, agl_ft=0.0, vs_fpm=0.0)
    assert f.phases() == ["approach", "landing", "rollout"]


def test_a_rejected_takeoff_goes_back_to_taxi():
    f = Flight()
    f.ramp(20, gs_to=70)
    f.ramp(20, gs_to=10).hold(10)
    assert f.phases() == ["taxi", "takeoff_roll", "taxi"]


def test_an_air_start_begins_where_it_is_not_in_an_initial_climb():
    assert Flight(on_ground=False, agl_ft=9000.0, gs_kt=250.0, gear=0.0).hold(5).phases() == [
        "cruise"
    ]
    descending = Flight(on_ground=False, agl_ft=9000.0, gs_kt=250.0, gear=0.0, vs_fpm=-600.0)
    assert descending.hold(5).phases() == ["descent"]


def test_vertical_speed_wandering_through_a_level_off_doesnt_flip_the_phase():
    f = Flight(on_ground=False, agl_ft=6000.0, gs_kt=200.0, gear=0.0)
    f.hold(10)
    for vs in (350, -320, 400, 250, -350, 310, 0, 0, 0):  # never 3 s the same way
        f.hold(1, vs_fpm=float(vs))
    assert f.phases() == ["cruise"]


def test_a_new_phase_is_dated_from_when_it_began():
    f = Flight(on_ground=False, agl_ft=6000.0, gs_kt=200.0, gear=0.0)
    f.hold(10).hold(10, vs_fpm=800.0)
    assert replay(f.samples) == [(0.0, "cruise"), (10.0, "climb")]


def test_no_on_ground_or_height_is_unknown():
    tracker = PhaseTracker()
    assert tracker.update(FlightSample(t=0.0, gs_kt=120.0)) is None
    assert tracker.phase == "unknown"
    assert tracker.update(FlightSample(t=1.0, agl_ft=3.0, gs_kt=0.0)) == "parked"  # height alone


def test_reset_starts_a_new_flight():
    tracker = PhaseTracker()
    for t in range(5):
        tracker.update(FlightSample(t=float(t), on_ground=True, agl_ft=0.0, gs_kt=0.0))
    assert tracker.phase == "parked"
    tracker.reset()  # an air start: the next sample is where it begins
    assert (
        tracker.update(FlightSample(t=5.0, on_ground=False, agl_ft=5000.0, vs_fpm=0.0)) == "cruise"
    )


@pytest.mark.parametrize(
    ("sample", "previous", "want"),
    [
        (dict(on_ground=True, gs_kt=50.0), "taxi", "takeoff_roll"),
        (dict(on_ground=True, gs_kt=50.0), "landing", "rollout"),
        (dict(on_ground=True, gs_kt=50.0), "rollout", "rollout"),
        (dict(on_ground=False, agl_ft=1800.0, vs_fpm=-700.0), "descent", "approach"),
        (dict(on_ground=False, agl_ft=1800.0, vs_fpm=0.0, flaps=0.3), "cruise", "approach"),
        (dict(on_ground=False, agl_ft=1800.0, vs_fpm=0.0), "cruise", "cruise"),  # low, clean, level
        # Clean and dipping from level flight (over hills): a descent first.
        (dict(on_ground=False, agl_ft=2380.0, vs_fpm=-311.0), "cruise", "descent"),
        (dict(on_ground=False, agl_ft=2380.0, vs_fpm=-311.0), None, "descent"),  # an air start
        (dict(on_ground=False, agl_ft=80.0, vs_fpm=-500.0), "approach", "landing"),
        (dict(on_ground=False, agl_ft=80.0, vs_fpm=900.0), "landing", "go_around"),
        # Above 1,000 ft a climb from an approach is a climb (an autopilot
        # levelling off a final-approach start at 1,500 ft).
        (dict(on_ground=False, agl_ft=1500.0, vs_fpm=700.0), "approach", "climb"),
        (dict(on_ground=False, agl_ft=800.0, vs_fpm=0.0), "initial_climb", "initial_climb"),
    ],
)
def test_candidate(sample, previous, want):
    assert candidate(FlightSample(t=0.0, **sample), previous) == want


def test_a_recorded_sample_reads_back():
    ev = {"t": 12.5, "kind": "flight", "phase": "taxi", "on_ground": True, "gs_kt": 12.0}
    s = sample_from_event(ev)
    assert s.t == 12.5 and s.on_ground is True and s.gs_kt == 12.0 and s.agl_ft is None


def test_dwell_is_a_few_seconds():
    assert 2.0 <= DWELL_SEC <= 5.0
