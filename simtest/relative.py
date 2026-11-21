"""Phase L.1's checklist: relative commands in the air, on each aircraft.

The air start and the hands-off set-up are Tier 1's (heading 180, level at
the aircraft's altitude). Each relative command is checked against the
target it should give from a state the steps before set up: a turn from the
heading the aircraft has settled on, a climb from the altitude it holds, a
speed change from the bug the autothrottle flies (jets) or from the airspeed
(pistons). The two questions are asked and answered with the absolute value,
which is always one of the two offered.
"""

from __future__ import annotations

from typing import Any

from .scenario import SimCheck, Step, equals, heading_near
from .tier1 import SETUPS, hands_off

AP = "sim/cockpit2/autopilot/"
BUG = AP + "heading_dial_deg_mag_pilot"
ALT_BUG = AP + "altitude_dial_ft"
SPEED_BUG = AP + "airspeed_dial_kts_mach"
IS_MACH = AP + "airspeed_is_mach"
IAS = "sim/cockpit2/gauges/indicators/airspeed_kts_pilot"
ALT = "sim/cockpit2/gauges/indicators/altitude_ft_pilot"
VS = "sim/cockpit2/gauges/indicators/vvi_fpm_pilot"
DONE_OR_SET = ("executed", "no_op")
# The loaded model must know this to run the checklist.
NEEDS_INTENT = "turn_by"
# A turn's reference is the heading flown, settled within this of the target.
SETTLED_DEG = 2.0
TURN_SEC = 90
# The aircraft's start altitudes (tier1.SETUPS), as said.
START_WORDS = {6000: "six thousand", 8000: "eight thousand", 10000: "ten thousand"}


def _bug(target: float, tol: float = 0.5) -> SimCheck:
    return equals(f"heading bug {target:03.0f}", BUG, target, tol=tol)


def _turn(name: str, say: str, from_deg: float, by: int) -> list[Step]:
    target = (from_deg + by) % 360 or 360
    side = "Left" if by < 0 else "Right"
    return [
        Step(
            name,
            say=say,
            reply=f"{side} {abs(by)}",
            # From the heading flown (settled within SETTLED_DEG of the last target).
            sim=(_bug(target, tol=SETTLED_DEG + 1), heading_near(target, SETTLED_DEG, TURN_SEC)),
        )
    ]


def _alt_bug_from_held(by: int) -> SimCheck:
    """Level in altitude hold: the bug is the altitude held (to 100 ft) plus
    the change."""

    def off(r: dict[str, Any]) -> float | None:
        if r.get(ALT_BUG) is None or r.get(ALT) is None:
            return None
        return round(float(r[ALT_BUG]) - (round(float(r[ALT]) / 100) * 100 + by))

    return SimCheck(
        f"altitude bug the altitude held {by:+d} ft",
        (ALT_BUG, ALT),
        lambda d: d is not None and abs(d) <= 100,
        value=off,
        expected="within 100 ft (the altitude is rounded to 100)",
    )


def _levelled_at_bug(within_sec: float) -> SimCheck:
    """Level within 150 ft of the altitude bug, whatever it was set to."""

    def level(r: dict[str, Any]) -> dict[str, int] | None:
        if any(r.get(k) is None for k in (ALT, VS, ALT_BUG)):
            return None
        return {
            "altitude": round(float(r[ALT])),
            "bug": round(float(r[ALT_BUG])),
            "vs": round(float(r[VS])),
        }

    return SimCheck(
        "levelled off at the bug",
        (ALT, VS, ALT_BUG),
        lambda v: v is not None and abs(v["altitude"] - v["bug"]) <= 150 and abs(v["vs"]) <= 300,
        value=level,
        within_sec=within_sec,
        expected="the bug ± 150 ft, |V/S| ≤ 300",
    )


def _speed_from_airspeed(by: int) -> SimCheck:
    """No autothrottle: the bug is the airspeed now (to 5 kt) plus the change."""

    def off(r: dict[str, Any]) -> float | None:
        if r.get(SPEED_BUG) is None or r.get(IAS) is None:
            return None
        return round(float(r[SPEED_BUG]) - float(r[IAS]) - by, 1)

    return SimCheck(
        f"speed bug the airspeed {by:+d} kt",
        (SPEED_BUG, IAS),
        lambda d: d is not None and abs(d) <= 6,
        value=off,
        expected="within 6 kt (the airspeed is rounded to 5)",
    )


def steps(aircraft: str) -> list[Step]:
    s = SETUPS[aircraft]
    alt = s.alt_ft
    out = hands_off(aircraft) + [
        Step(
            "settled on 180",
            say="heading one eight zero",
            outcomes=DONE_OR_SET,
            sim=(_bug(180), heading_near(180, SETTLED_DEG, TURN_SEC)),
        ),
    ]
    out += _turn("turn left twenty", "turn left twenty degrees", 180, -20)
    out += _turn("come right thirty", "come right thirty degrees", 160, 30)
    out += [
        # 190 + 120 = 310, or the heading 120: asked, answered with the heading.
        Step(
            "heading or turn is asked",
            say="turn right one two zero",
            outcomes=("ambiguous",),
            reply="Heading one two zero or",
            vimaan_checks=False,
            sim=(_bug(190, tol=SETTLED_DEG + 1),),  # nothing set yet
        ),
        Step("answered: heading 120", say="one two zero", sim=(_bug(120),)),
        Step(
            "compass: turn south",
            say="turn south",
            sim=(_bug(180), heading_near(180, SETTLED_DEG + 3, TURN_SEC)),
        ),
        # Level in altitude hold: a change is from the altitude held, to 100
        # ft. That is where altitude hold caught the aircraft after the air
        # start, not the start altitude: the first run (2026-11-20) found
        # them 100-500 ft off it. The bug moves, the aircraft stays: nothing
        # climbs (the Baron and King Air have no autothrottle, and a V/S climb
        # with no one on the throttle stalled them both).
        Step(
            "climb another thousand",
            say="climb another thousand",
            reply="Up 1000",
            sim=(_alt_bug_from_held(1000),),
        ),
        Step(
            "altitude or change is asked",
            say="descend two thousand feet",
            outcomes=("ambiguous",),
            reply="Altitude 2000 or",
            vimaan_checks=False,
            no_actions=True,
        ),
        Step(
            "answered: 2000", say="two thousand", sim=(equals("altitude bug 2000", ALT_BUG, 2000),)
        ),
        Step(
            "down five hundred",
            say="down five hundred feet",
            reply="Down 500",
            sim=(_alt_bug_from_held(-500),),
        ),
        # Flown down to it, as Tier 1's V/S descent: no power needed.
        Step(
            "descend to it",
            say="descend at one thousand five hundred feet per minute",
            reply="feet per minute, to",
            sim=(_levelled_at_bug(within_sec=500 / 1500 * 60 + 120),),
        ),
        Step(
            f"back to {alt}",
            say=f"altitude {START_WORDS[alt]}",
            outcomes=DONE_OR_SET,
            sim=(equals(f"altitude bug {alt}", ALT_BUG, alt, tol=0.5),),
        ),
    ]
    if s.jet:  # the autothrottle flies the bug: a change is from the bug
        out += [
            Step(
                f"speed {s.speed}",
                say=f"speed {s.speed_words}",
                outcomes=DONE_OR_SET,
                sim=(equals(f"speed bug {s.speed}", SPEED_BUG, s.speed, tol=1),),
            ),
            Step(
                "speed up ten",
                say="speed up ten knots",
                reply="10 knots faster",
                sim=(equals(f"speed bug {s.speed + 10}", SPEED_BUG, s.speed + 10, tol=1),),
            ),
            Step(
                "slow down twenty",
                say="slow down twenty knots",
                reply="20 knots slower",
                sim=(equals(f"speed bug {s.speed - 10}", SPEED_BUG, s.speed - 10, tol=1),),
            ),
            Step(
                "mach .78",
                say="mach point seven eight",
                outcomes=DONE_OR_SET,
                sim=(equals("dial in Mach", IS_MACH, 1, within_sec=3),),
            ),
            Step(
                "a change in Mach is refused",
                say="speed up ten knots",
                outcomes=("rejected",),
                reply="Mach",
                vimaan_checks=False,
                sim=(equals("still Mach", IS_MACH, 1),),
            ),
            Step(
                f"back to {s.speed} kt",
                say=f"speed {s.speed_words}",
                outcomes=DONE_OR_SET,
                sim=(equals("dial in knots", IS_MACH, 0, within_sec=3),),
            ),
        ]
    else:
        out += [
            Step(
                "speed up ten",
                say="speed up ten knots",
                reply="10 knots faster",
                sim=(_speed_from_airspeed(10),),
            ),
        ]
    return out
