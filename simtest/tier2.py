"""The Tier 2 checklist (PHASE_PLAN, Tier 2) as scenarios: radios, course and
transponder, flown in the air like Tier 1, with the autopilot flying.

Needs a model trained on Tier 2 (v20 or later): the runner refuses to start
on an older one.

The transponder is checked the way each aircraft shows it. X-Plane's mode
enum is the answer on the Baron and King Air. The 737's selector is checked by
its position, because X-Plane reads mode 2 for every position from ALT RPTG
OFF to TA/RA. The A330's script sets X-Plane's mode from its three selectors:
with altitude reporting on in flight that is TA (6) or TA/RA (7), never 3.
"""

from __future__ import annotations

from .scenario import SimCheck, Step, equals
from .tier1 import DONE_OR_SET, SETUPS, hands_off

RADIO = "sim/cockpit2/radios/actuators/"
XPDR_MODE = RADIO + "transponder_mode"
XPDR_CODE = RADIO + "transponder_code"
XPDR_IDENT = "sim/cockpit2/radios/indicators/transponder_id"
B738_SELECTOR = "laminar/B738/knob/transponder_pos"
# "pending": the 737 and A330 turn their selectors over several frames.
SET_OR_TURNING = (*DONE_OR_SET, "pending")
# X-Plane's modes that reply: on (A), alt (C), ground, TA, TA/RA.
REPLYING = (2, 3, 5, 6, 7)
# The intent a model must know for this checklist.
NEEDS_INTENT = "set_transponder_code"


def _mode_in(what: str, modes: tuple[int, ...]) -> SimCheck:
    return SimCheck(
        what,
        (XPDR_MODE,),
        lambda v: v is not None and int(v) in modes,
        within_sec=5,
        expected="mode " + " / ".join(str(m) for m in modes),
    )


def _altitude_reporting(aircraft: str) -> SimCheck:
    if aircraft == "737":
        return SimCheck(
            "selector XPNDR, TA or TA/RA",
            (B738_SELECTOR,),
            lambda v: v is not None and 3 <= round(float(v)) <= 5,
            within_sec=5,
            expected="position 3-5",
        )
    if aircraft == "a330":
        return _mode_in("TA or TA/RA (altitude reported)", (6, 7))
    return _mode_in("mode C", (3,))


def _transponder(aircraft: str) -> list[Step]:
    has_off = aircraft not in ("737", "a330")
    return [
        Step(
            "transponder alt",
            say="transponder alt",
            outcomes=SET_OR_TURNING,
            sim=(_altitude_reporting(aircraft),),
        ),
        Step(
            "squawk standby",
            say="squawk standby",
            outcomes=SET_OR_TURNING,
            sim=(_mode_in("standby", (1,)),),
        ),
        Step(
            "ident refused in standby",
            say="ident",
            outcomes=("rejected",),
            reply="standby",
            vimaan_checks=False,
            sim=(equals("not identing", XPDR_IDENT, 0),),
        ),
        Step(
            "transponder on",
            say="transponder on",
            outcomes=SET_OR_TURNING,
            sim=(_mode_in("replying", REPLYING),),
        ),
        Step(
            "ident",
            say="ident",
            sim=(equals("identing", XPDR_IDENT, 1, within_sec=3),),
        ),
        Step(
            "transponder off" if has_off else "transponder off refused (no off)",
            say="transponder off",
            outcomes=DONE_OR_SET if has_off else ("rejected",),
            reply=None if has_off else "no off",
            vimaan_checks=has_off,
            sim=(_mode_in("off", (0,)),) if has_off else (_mode_in("still replying", REPLYING),),
        ),
        # Leave it reporting altitude, as it would be flown.
        Step(
            "transponder altitude",
            say="transponder altitude",
            outcomes=SET_OR_TURNING,
            sim=(_altitude_reporting(aircraft),),
        ),
    ]


def steps(aircraft: str) -> list[Step]:
    if aircraft not in SETUPS:
        raise KeyError(aircraft)
    out = hands_off(aircraft)
    out += [
        # COM: standby, then flip it to active.
        Step(
            "COM 1 standby 121.9",
            say="com one standby one two one decimal niner",
            outcomes=DONE_OR_SET,
            sim=(equals("COM 1 standby 121.9", RADIO + "com1_standby_frequency_hz_833", 121900),),
        ),
        Step(
            "flip COM 1",
            say="flip com one",
            sim=(
                equals("COM 1 active 121.9", RADIO + "com1_frequency_hz_833", 121900, within_sec=3),
            ),
        ),
        Step(
            "COM 2 standby 132.45",
            say="com two standby one three two decimal four five",
            outcomes=DONE_OR_SET,
            sim=(equals("COM 2 standby 132.45", RADIO + "com2_standby_frequency_hz_833", 132450),),
        ),
        # NAV 1 for an ILS: frequency and course.
        Step(
            "NAV 1 110.50",
            say="tune nav one to one one zero decimal five",
            outcomes=DONE_OR_SET,
            sim=(equals("NAV 1 110.50", RADIO + "nav1_frequency_hz", 11050),),
        ),
        Step(
            "course 090",
            say="course zero niner zero",
            outcomes=DONE_OR_SET,
            sim=(equals("NAV 1 course 090", RADIO + "nav1_obs_deg_mag_pilot", 90, tol=0.5),),
        ),
        # NAV 2: standby, flip, course.
        Step(
            "NAV 2 standby 113.9",
            say="nav two standby one one three decimal niner",
            outcomes=DONE_OR_SET,
            sim=(equals("NAV 2 standby 113.9", RADIO + "nav2_standby_frequency_hz", 11390),),
        ),
        Step(
            "flip NAV 2",
            say="swap nav two",
            sim=(equals("NAV 2 active 113.9", RADIO + "nav2_frequency_hz", 11390, within_sec=3),),
        ),
        Step(
            "NAV 2 course 180",
            say="nav two course one eight zero",
            outcomes=DONE_OR_SET,
            sim=(equals("NAV 2 course 180", RADIO + "nav2_obs_deg_mag_pilot", 180, tol=0.5),),
        ),
        # Squawk codes, a leading zero, and an emergency code held for
        # confirmation, then cancelled.
        Step(
            "squawk 4217",
            say="squawk four two one seven",
            outcomes=DONE_OR_SET,
            sim=(equals("code 4217", XPDR_CODE, 4217),),
        ),
        Step(
            "squawk 0120",
            say="squawk zero one two zero",
            outcomes=DONE_OR_SET,
            sim=(equals("code 0120", XPDR_CODE, 120),),
            note="X-Plane keeps the code's digits as a number: 0120 is 120",
        ),
        Step(
            "squawk 1200",
            say="squawk one two zero zero",
            outcomes=DONE_OR_SET,
            sim=(equals("code 1200", XPDR_CODE, 1200),),
        ),
        Step(
            "emergency squawk is held",
            say="squawk seven seven zero zero",
            outcomes=("safety_hold",),
            reply="Confirm emergency squawk",
            vimaan_checks=False,
            sim=(equals("code still 1200", XPDR_CODE, 1200),),
        ),
        Step(
            "cancel the emergency squawk",
            say="cancel",
            outcomes=("cancelled",),
            vimaan_checks=False,
            after="emergency squawk is held",
            sim=(equals("code still 1200", XPDR_CODE, 1200, within_sec=2),),
        ),
        *_transponder(aircraft),
    ]
    return out
