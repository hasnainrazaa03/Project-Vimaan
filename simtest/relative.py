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

from .scenario import SimCheck, Step, equals, heading_near, levelled_at
from .tier1 import SETUPS, THOUSANDS, hands_off

AP = "sim/cockpit2/autopilot/"
BUG = AP + "heading_dial_deg_mag_pilot"
ALT_BUG = AP + "altitude_dial_ft"
SPEED_BUG = AP + "airspeed_dial_kts_mach"
IS_MACH = AP + "airspeed_is_mach"
IAS = "sim/cockpit2/gauges/indicators/airspeed_kts_pilot"
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
    alt, up = s.alt_ft, s.alt_ft + 1000
    down = s.alt_ft - 1000  # in tier1.THOUSANDS for every aircraft
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
        # Altitude hold at the start altitude: a change is from where we are.
        Step(
            "climb another thousand",
            say="climb another thousand",
            reply="Up 1000",
            sim=(equals(f"altitude bug {up}", ALT_BUG, up, tol=0.5),),
        ),
        Step(
            "climb to it",
            say="climb at one thousand feet per minute",
            reply=f"to {up}",
            sim=(levelled_at(up, within_sec=1000 / 1000 * 60 + 120),),
        ),
        Step(
            "altitude or change is asked",
            say="descend two thousand feet",
            outcomes=("ambiguous",),
            reply=f"Altitude 2000 or {down} feet",
            vimaan_checks=False,
            sim=(equals("altitude bug unchanged", ALT_BUG, up, tol=0.5),),
        ),
        Step(
            f"answered: {down}",
            say=THOUSANDS[down],
            sim=(equals(f"altitude bug {down}", ALT_BUG, down, tol=0.5),),
        ),
        # Still level at `up` in altitude hold: from where we are, to 100 ft.
        Step(
            "down five hundred",
            say="down five hundred feet",
            reply="Down 500",
            sim=(equals(f"altitude bug {up - 500}", ALT_BUG, up - 500, tol=100),),
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
