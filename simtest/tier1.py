"""The Tier 1 checklist (PHASE_PLAN, T1 Part C) as scenarios, one per
aircraft, flown in the air west of Portland with no one at the controls.

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
FLAPS = "sim/cockpit2/controls/flap_handle_request_ratio"
IAS = "sim/cockpit2/gauges/indicators/airspeed_kts_pilot"
VFE = "sim/aircraft/view/acf_Vfe"
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
    autothrottle: str  # "none" (refused), "generic" (737) or "airbus" (A/THR)
    # Thrust levers set by the pilot before the autothrottle (A330: CL detent,
    # _fadec_tla 0.53; A/THR stays armed at FLX/MCT 0.76 and TOGA 1.0).
    levers: float | None = None
    athr_on_from: int = 1  # autothrottle_enabled from this counts as on (A330: armed, 0)


SETUPS: dict[str, Setup] = {
    "baron": Setup("BE58", 6000, 150, 4000, "four thousand", 160, "one six zero", False, "none"),
    "kingair": Setup("BE9L", 8000, 180, 6000, "six thousand", 160, "one six zero", False, "none"),
    "737": Setup(
        "B738",
        10000,
        250,
        8000,
        "eight thousand",
        250,
        "two five zero",
        True,
        "generic",
    ),
    "a330": Setup(
        "A333",
        10000,
        250,
        8000,
        "eight thousand",
        250,
        "two five zero",
        True,
        "airbus",
        levers=0.53,
        athr_on_from=0,
    ),
}


def _faster_than_vfe(want: bool, vfe_full: float | None) -> SimCheck:
    """Above (or below) the full-flap limit, clear of the line. That limit
    comes from the .acf (``vfe_full``); X-Plane's acf_Vfe is the first notch's."""

    def margin(r: dict) -> float | None:
        limit = vfe_full or r.get(VFE)
        if r.get(IAS) is None or not limit:
            return None
        return round(float(r[IAS]) - float(limit), 1)

    return SimCheck(
        "above full-flap speed" if want else "below full-flap speed",
        (IAS, VFE),
        lambda m: m is not None and (m > 5 if want else m < -5),
        value=margin,
        expected="IAS − Vfe(full) > 5" if want else "IAS − Vfe(full) < −5",
    )


THOUSANDS = {5000: "five thousand", 7000: "seven thousand", 9000: "nine thousand"}
THROTTLE = "sim/cockpit2/engine/actuators/throttle_ratio_all"
ATHR = AP + "autothrottle_enabled"


def _athr(s: Setup, on: bool) -> SimCheck:
    return SimCheck(
        "autothrottle on" if on else "autothrottle off",
        (ATHR,),
        (lambda v: v is not None and v >= s.athr_on_from)
        if on
        else (lambda v: v is not None and v < s.athr_on_from),
        within_sec=4,  # the 737 arms its switch, then engages 0.5 s later
        expected=f"≥ {s.athr_on_from}" if on else f"< {s.athr_on_from}",
    )


def _manage_thrust(s: Setup) -> list[Step]:
    """Jets: thrust managed from the start. Unmanaged, the A330 left at its
    air-start thrust reached 337 kt (Vmo 330) and its high-speed protection
    took the autopilot (re-run, 2026-11-12)."""
    steps = []
    if s.levers is not None:
        steps.append(
            Step(
                "thrust levers to CL (pilot)",
                write=(THROTTLE, s.levers),
                sim=(equals("levers at CL", THROTTLE, s.levers, tol=0.02, within_sec=2),),
            )
        )
    sim = [_athr(s, True)]
    steps.append(
        Step(
            "autothrottle on",
            say="auto throttle on",
            outcomes=DONE_OR_SET,
            sim=tuple(sim),
            note="the 737 arms its A/T switch first if it's off",
        )
    )
    return steps


def _ias_near(kt: float, tol: float, within_sec: float) -> SimCheck:
    return SimCheck(
        f"airspeed settles at {kt:.0f}",
        (IAS,),
        lambda v: v is not None and abs(float(v) - kt) <= tol,
        value=lambda r: None if r.get(IAS) is None else round(float(r[IAS])),
        within_sec=within_sec,
        expected=f"{kt:.0f} ± {tol:.0f} kt",
    )


def _baro(words: str, inhg: float, within: float = 35.0) -> Step:
    # The 737 turns its knob one click per frame: allow time.
    return Step(
        words,
        say=words,
        outcomes=(*DONE_OR_SET, "pending"),  # "pending": the 737 turns its knob over frames
        sim=(equals(f"altimeter {inhg}", BARO, inhg, tol=0.012, within_sec=within),),
    )


def hands_off(aircraft: str) -> list[Step]:
    """The aircraft flies itself: autopilot on, thrust managed (jets),
    heading mode and altitude hold. Every checklist starts with these."""
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
    ]
    if s.jet:
        out[1:1] = _manage_thrust(s)  # straight after "autopilot on"
    return out


def steps(aircraft: str, vfe_full: float | None = None) -> list[Step]:
    """``vfe_full``: the full-flap limit from the aircraft file
    (simtest.aircraft_file.flap_limits), if known."""
    s = SETUPS[aircraft]
    out = hands_off(aircraft) + [
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
            sim=(equals(f"speed bug {s.speed}", DIAL_SPEED, s.speed, tol=1),)
            # With the autothrottle on, the aircraft has to get there.
            + ((_ias_near(s.speed, 20, 120),) if s.jet else ()),
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
        # Somewhere to go first: at the selected altitude an Airbus doesn't
        # change mode (re-run 2026-11-12: "Level change mode did not respond").
        Step(
            f"altitude {s.descend_to + 1000} for level change",
            say=f"altitude {THOUSANDS[s.descend_to + 1000]}",
            outcomes=DONE_OR_SET,
            sim=(equals("altitude bug", AP + "altitude_dial_ft", s.descend_to + 1000, tol=1),),
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
    else:  # off, then on again, so the speed stays managed for the rest
        out += [
            Step(
                "autothrottle off",
                say="auto throttle off",
                outcomes=DONE_OR_SET,
                sim=(_athr(s, False),),
            ),
            Step(
                "autothrottle on again",
                say="auto throttle on",
                outcomes=DONE_OR_SET,
                sim=(_athr(s, True),),
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
        # The flap-speed rule follows X-Plane's acf_Vfe for this aircraft:
        # held above it, done below it (then put back up).
        Step(
            "flaps full above Vfe is held",
            say="flaps full",
            outcomes=("safety_hold",),
            reply="Confirm flaps",
            vimaan_checks=False,
            only_if=_faster_than_vfe(True, vfe_full),
        ),
        Step(
            "cancel the flaps",
            say="cancel",
            outcomes=("cancelled",),
            vimaan_checks=False,
            after="flaps full above Vfe is held",
            sim=(equals("flaps still up", FLAPS, 0, tol=0.01, within_sec=2),),
        ),
        Step(
            "flaps full below Vfe",
            say="flaps full",
            outcomes=DONE,
            vimaan_checks=False,
            only_if=_faster_than_vfe(False, vfe_full),
            sim=(equals("flaps full", FLAPS, 1, tol=0.01, within_sec=3),),
        ),
        Step(
            "flaps up again",
            say="flaps up",
            outcomes=DONE,
            vimaan_checks=False,
            after="flaps full below Vfe",
            sim=(equals("flaps up", FLAPS, 0, tol=0.01, within_sec=3),),
        ),
        Step(
            "autopilot off, no callout",
            say="autopilot off",
            outcomes=DONE,
            sim=(equals("servos off", SERVOS, 0, within_sec=3),),
            no_callout="Autopilot disconnected",
        ),
        # Straight after "autopilot on": until #38, Vimaan took a disconnect
        # within 5 s of its own "autopilot on" as its doing, and stayed quiet.
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
