"""Phase L.4.6: the six checklists in the sim, on each aircraft.

Two runs, each started the way its checklists are used:

- **On the runway** (``checklists-ground``, the takeoff's start: engines
  running, parked): before takeoff, then after landing (which undoes it),
  each with every offer confirmed and then asked again; the approach with
  one offer declined and one left to run out; after takeoff, where "Raise
  the gear?" confirmed on the ground is held by the gear-up rule and the
  hold cancelled; before landing stopped by "cancel"; and the shutdown,
  engines and all, then asked again.
- **In the air** (``checklists-air``, Tier 1's air start, hands off): after
  takeoff, the approach and before landing, every offer confirmed, the last
  two asked again. The gear comes down only below its limit speed (else the
  offer is declined); after takeoff, stopped by "cancel".

Nothing is aircraft by aircraft: what each run must show follows from what
Vimaan offered and how it was answered (``scenario.Follow``). An item set is
read again from the sim where that is a thing of its own (the gear, the
flaps, the parking brake, the speed brakes, the engines), and asked again,
every item set must read right.
"""

from __future__ import annotations

from typing import Any

from .flights import PLANES
from .scenario import Follow, SimCheck, Step, above, below, equals, phase_is
from .tier1 import hands_off

# The loaded model must know this to run the checklists (v28).
NEEDS_INTENT = "run_checklist"

GEAR = "sim/flightmodel2/gear/deploy_ratio[0]"
FLAPS = "sim/cockpit2/controls/flap_handle_request_ratio"
PARK = "sim/flightmodel/controls/parkbrake"
SPEEDBRAKE = "sim/cockpit2/controls/speedbrake_ratio"  # armed: below 0
RUNNING = "sim/flightmodel/engine/ENGN_running"
IAS = "sim/cockpit2/gauges/indicators/airspeed_kts_pilot"
VLE = "sim/aircraft/overflow/acf_Vle"  # the gear's limit speed, extended
# The gear comes down in the air only this far below its limit speed.
VLE_MARGIN_KT = 10.0

GEAR_DOWN = above("gear down", GEAR, 0.99, within_sec=30)
FLAPS_OUT = above("flaps out", FLAPS, 0.05, within_sec=20)
FLAPS_UP = below("flaps up", FLAPS, 0.01, within_sec=30)
PARKED = above("parking brake set", PARK, 0.5, within_sec=5)
RETRACTED = equals("speed brakes retracted", SPEEDBRAKE, 0.0, tol=0.01, within_sec=5)
ARMED = below("speed brakes armed", SPEEDBRAKE, -0.01, within_sec=5)


def _stopped(engines: int) -> SimCheck:
    refs = tuple(f"{RUNNING}[{i}]" for i in range(engines))
    return SimCheck(
        "engines stopped",
        refs,
        lambda v: v == (0,) * engines,
        value=lambda r: tuple(r.get(x) for x in refs),
        within_sec=60,
        expected="all stopped",
    )


def gear_below_its_speed(bridge: Any) -> str:
    """The gear's offer: confirmed below its limit speed, else declined."""
    r = bridge.read(IAS, VLE)
    ias, vle = r.get(IAS), r.get(VLE)
    if ias is None or not vle:
        return "cancel"
    return "confirm" if float(ias) <= float(vle) - VLE_MARGIN_KT else "cancel"


def _start(name: str, follow: Follow, step: str = "", **kw: Any) -> Step:
    title = f"{name} checklist"
    return Step(
        step or title,
        say=title,
        reply=title.capitalize(),
        vimaan_checks=False,  # an item's own read-backs are the offer's command's
        follow=follow,
        **kw,
    )


def ground(aircraft: str, hands: Any = None) -> list[Step]:
    """On the runway, engines running (``hands``: the flights' runner,
    unused: nothing is flown)."""
    p = PLANES[aircraft]
    return [
        Step("parked on the runway", sim=(phase_is("parked", within_sec=20),)),
        _start(
            "before takeoff",
            Follow("before takeoff", sets={"flaps": FLAPS_OUT, "speed brakes": RETRACTED}),
        ),
        _start("before takeoff", Follow("before takeoff", again=True), "before takeoff again"),
        _start(
            "after landing",
            Follow("after landing", sets={"flaps": FLAPS_UP, "speed brakes": RETRACTED}),
        ),
        _start("after landing", Follow("after landing", again=True), "after landing again"),
        _start(
            "approach",
            Follow("approach", answers={"landing lights": "cancel", "auto brake": "wait"}),
            "approach, one declined, one not answered",
        ),
        _start(
            "after takeoff",
            Follow("after takeoff", held=("gear",)),
            "after takeoff on the ground, the gear held",
            sim=(equals("gear still down", GEAR, 1.0, tol=0.01),),
        ),
        _start("before landing", Follow("before landing", stop=True), "before landing cancelled"),
        _start(
            "shutdown",
            Follow("shutdown", sets={"parking brake": PARKED, "engines": _stopped(p.engines)}),
        ),
        _start("shutdown", Follow("shutdown", again=True), "shutdown again"),
    ]


def air(aircraft: str) -> list[Step]:
    """In the air, the aircraft flying itself."""
    gear = {"gear": gear_below_its_speed}
    return hands_off(aircraft) + [
        _start("after takeoff", Follow("after takeoff")),
        _start("approach", Follow("approach")),
        _start("approach", Follow("approach", again=True), "approach again"),
        _start(
            "before landing",
            Follow("before landing", answers=gear, sets={"gear": GEAR_DOWN, "speed brakes": ARMED}),
        ),
        _start(
            "before landing",
            Follow("before landing", answers=gear, again=True),
            "before landing again",
        ),
        _start("after takeoff", Follow("after takeoff", stop=True), "after takeoff cancelled"),
    ]
