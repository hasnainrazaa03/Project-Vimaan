"""The Tier 1 checklist (PHASE_PLAN, T1 Part C) as scenarios, one per
aircraft, flown in the air over Seattle with no one at the controls.

Ground items (engine start, parking brake, takeoff gear rule) wait for A3.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scenario import SimCheck, Step, equals, heading_near, levelled_at

AP = "sim/cockpit2/autopilot/"
SERVOS = AP + "servos_on"
BARO = "sim/cockpit2/gauges/actuators/barometer_setting_in_hg_pilot"
DIAL_SPEED = AP + "airspeed_dial_kts_mach"
IS_MACH = AP + "airspeed_is_mach"
DONE = ("executed",)
DONE_OR_SET = ("executed", "no_op")  # "Heading already on" is fine too


@dataclass(frozen=True)
class Setup:
    """Where and how each aircraft starts, and what it has."""

    icao: str
    alt_ft: int
    kt: int
    descend_to: int
    descend_to_words: str
    speed: int
    speed_words: str
    jet: bool
    autothrottle: str  # "none" (refused), "generic" (737) or "airbus" (A/THR, lever-dependent)


SETUPS: dict[str, Setup] = {
    "baron": Setup("BE58", 6000, 150, 4000, "four thousand", 160, "one six zero", False, "none"),
    "kingair": Setup("BE9L", 8000, 180, 6000, "six thousand", 160, "one six zero", False, "none"),
    "737": Setup("B738", 10000, 250, 8000, "eight thousand", 250, "two five zero", True, "generic"),
    "a330": Setup("A333", 10000, 250, 8000, "eight thousand", 250, "two five zero", True, "airbus"),
}


def _baro(words: str, inhg: float, within: float = 35.0) -> Step:
    # The 737 turns its knob one click per frame: allow time.
    return Step(
        words,
        say=words,
        outcomes=DONE_OR_SET,
        sim=(equals(f"altimeter {inhg}", BARO, inhg, tol=0.012, within_sec=within),),
    )


def steps(aircraft: str) -> list[Step]:
    s = SETUPS[aircraft]
    out = [
        Step(
            "autopilot on",
            say="autopilot on",
            outcomes=DONE_OR_SET,
            sim=(equals("servos on", SERVOS, 1, within_sec=3),),
        ),
        Step(
            "heading mode",
            say="heading mode",
            outcomes=DONE_OR_SET,
            sim=(equals("heading mode on", AP + "heading_status", 2, within_sec=3),),
        ),
        Step(
            "altitude hold",
            say="altitude hold",
            outcomes=DONE_OR_SET,
            sim=(equals("altitude hold on", AP + "altitude_hold_status", 2, within_sec=3),),
        ),
        Step(
            "heading 270",
            say="heading two seven zero",
            sim=(
                equals("heading bug 270", AP + "heading_dial_deg_mag_pilot", 270, tol=0.5),
                heading_near(270, tol=15, within_sec=100),
            ),
        ),
        Step(
            f"speed {s.speed}",
            say=f"speed {s.speed_words}",
            outcomes=DONE_OR_SET,
            sim=(equals(f"speed bug {s.speed}", DIAL_SPEED, s.speed, tol=1),),
        ),
    ]
    if s.jet:
        out += [
            Step(
                "mach .78",
                say="mach point seven eight",
                outcomes=DONE_OR_SET,
                sim=(
                    equals("dial in Mach", IS_MACH, 1, within_sec=3),
                    equals("Mach .78", DIAL_SPEED, 0.78, tol=0.005, within_sec=3),
                ),
            ),
            Step(
                f"back to {s.speed} kt",
                say=f"speed {s.speed_words}",
                outcomes=DONE_OR_SET,
                sim=(
                    equals("dial in knots", IS_MACH, 0, within_sec=3),
                    equals(f"speed bug {s.speed}", DIAL_SPEED, s.speed, tol=1, within_sec=3),
                ),
            ),
        ]
    climb_sec = abs(s.alt_ft - s.descend_to) / 1500 * 60 + 120
    out += [
        Step(
            f"altitude {s.descend_to}",
            say=f"altitude {s.descend_to_words}",
            outcomes=DONE_OR_SET,
            sim=(
                equals(
                    f"altitude bug {s.descend_to}", AP + "altitude_dial_ft", s.descend_to, tol=1
                ),
            ),
        ),
        Step(
            "V/S descent and level-off",
            say="descend at one thousand five hundred feet per minute",
            reply=f"to {s.descend_to}",
            sim=(levelled_at(s.descend_to, within_sec=climb_sec),),
            note="the main Tier 1 check: the aircraft levels off by itself",
        ),
        Step("level change", say="level change", outcomes=DONE_OR_SET),
        Step(
            "altitude hold again",
            say="altitude hold",
            outcomes=DONE_OR_SET,
            sim=(equals("altitude hold on", AP + "altitude_hold_status", 2, within_sec=5),),
        ),
        Step("nav mode", say="nav mode", outcomes=DONE_OR_SET),
        Step("nav mode off", say="nav mode off", outcomes=DONE_OR_SET),
        Step("approach mode", say="approach mode", outcomes=DONE_OR_SET),
        Step("approach mode off", say="approach mode off", outcomes=DONE_OR_SET),
        Step(
            "heading mode again",
            say="heading mode",
            outcomes=DONE_OR_SET,
            sim=(equals("heading mode on", AP + "heading_status", 2, within_sec=3),),
        ),
    ]
    if s.autothrottle == "none":
        out.append(
            Step(
                "autothrottle refused",
                say="auto throttle on",
                outcomes=("rejected",),
                reply="no autothrottle",
                vimaan_checks=False,
            )
        )
    elif s.autothrottle == "generic":
        out += [
            Step(
                "autothrottle on",
                say="auto throttle on",
                outcomes=DONE_OR_SET,
                sim=(
                    SimCheck(
                        "autothrottle engaged",
                        (AP + "autothrottle_enabled",),
                        lambda v: v is not None and v >= 1,
                        within_sec=3,
                        expected="≥ 1",
                    ),
                ),
            ),
            Step(
                "autothrottle off",
                say="auto throttle off",
                outcomes=DONE_OR_SET,
                sim=(
                    SimCheck(
                        "autothrottle off",
                        (AP + "autothrottle_enabled",),
                        lambda v: v is not None and v <= 0,
                        within_sec=3,
                        expected="≤ 0",
                    ),
                ),
            ),
        ]
    else:  # Airbus A/THR: what it does depends on the thrust levers (AIRCRAFT_PROFILES)
        out += [
            Step(
                "A/THR on",
                say="auto throttle on",
                outcomes=("executed", "no_op", "rejected", "failed"),
                vimaan_checks=False,
                note="depends on the thrust levers; recorded, not judged",
            ),
            Step(
                "A/THR off",
                say="auto throttle off",
                outcomes=("executed", "no_op", "rejected", "failed"),
                vimaan_checks=False,
                note="depends on the thrust levers; recorded, not judged",
            ),
        ]
    out += [
        _baro("altimeter three zero one five", 30.15),
        _baro("altimeter two niner niner two", 29.92),
        _baro("q n h one zero one three", 29.91),
        _baro("altimeter standard", 29.92, within=10),
        Step(
            "COM 2 by frequency first",
            say="frequency one three three decimal one on com two",
            outcomes=DONE_OR_SET,
            sim=(
                equals(
                    "COM 2 133.1", "sim/cockpit2/radios/actuators/com2_frequency_hz_833", 133100
                ),
            ),
        ),
        Step(
            "COM 1",
            say="tune com one to one two one decimal five",
            outcomes=DONE_OR_SET,
            sim=(
                equals(
                    "COM 1 121.5", "sim/cockpit2/radios/actuators/com1_frequency_hz_833", 121500
                ),
            ),
        ),
        Step(
            "gear up (already up)",
            say="landing gear up",
            outcomes=DONE_OR_SET,
            sim=(equals("gear handle up", "sim/cockpit2/controls/gear_handle_down", 0),),
        ),
        Step(
            "flaps full above Vfe is held",
            say="flaps full",
            outcomes=("safety_hold",),
            reply="Confirm flaps",
            vimaan_checks=False,
        ),
        Step(
            "cancel the flaps",
            say="cancel",
            outcomes=("cancelled",),
            vimaan_checks=False,
            sim=(
                equals(
                    "flaps still up",
                    "sim/cockpit2/controls/flap_handle_request_ratio",
                    0,
                    tol=0.01,
                    within_sec=2,
                ),
            ),
        ),
        Step(
            "autopilot off, no callout",
            say="autopilot off",
            outcomes=DONE,
            sim=(equals("servos off", SERVOS, 0, within_sec=3),),
            no_callout="Autopilot disconnected",
        ),
        Step(
            "autopilot on again",
            say="autopilot on",
            outcomes=DONE,
            sim=(equals("servos on", SERVOS, 1, within_sec=3),),
        ),
        Step(
            "pilot disconnect is called out",
            command="sim/autopilot/servos_fdir_off",
            sim=(equals("servos off", SERVOS, 0, within_sec=3),),
            callout="Autopilot disconnected",
        ),
    ]
    return out
