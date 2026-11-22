"""The Tier 3 checklist (PHASE_PLAN, Tier 3) as scenarios: lights, cabin
signs, yaw damper, anti-ice, electrical, fuel pumps, speed brake and
autobrake, flown in the air like Tier 1 and 2, with the autopilot flying.

Needs a model trained on Tier 3 (v26 or later): the runner refuses to start on
an older one.

Each switch is checked the way its aircraft shows it
(docs/AIRCRAFT_PROFILES.md, "Tier 3 findings"): the 737's own switch
positions, the A330's buttons, the King Air's selectors, X-Plane's values
on the Baron. Where an aircraft has no such switch, the step expects
Vimaan to say so. Turning power, fuel pumps or pitot heat off in the air is
held for confirmation, then cancelled; nothing is switched off for real
(the A330's probe heat has no off: confirmed, it asks "on or auto?").
"""

from __future__ import annotations

import dataclasses

from .scenario import SimCheck, Step, equals
from .tier1 import DONE_OR_SET, SETUPS, hands_off

SW = "sim/cockpit2/switches/"
ICE = "sim/cockpit2/ice/"
# No generated voice says "pitot" so the recogniser hears it ("radio heat
# on", "feet heat on", "zero heat on", whatever the spelling). "probe" is
# the same slot value; "pitot" itself is left to the owner's voice check.
PITOT_SAY = "probe heat on"
PITOT_OFF_SAY = "probe heat off"
# Two of four generated voices hear "prop heat" as "probe heat" (the pitot
# heat), "propeller" never. "prop" is left to the owner's voice check too.
PROP_SAY = {"on": "propeller heat on", "off": "propeller heat off"}
EL = "sim/cockpit2/electrical/"
# Most Tier 3 settings take a few frames (steps wait for each switch).
SET_OR_TURNING = (*DONE_OR_SET, "pending")
# The intent a model must know for this checklist.
NEEDS_INTENT = "set_autobrake"

B738 = "laminar/B738/toggle_switch/"
A333 = "laminar/A333/"
C90 = "laminar/c90/"
CITX = "laminar/CitX/"
# Phase L.2's default fleet: what each cockpit has (docs/AIRCRAFT_PROFILES.md,
# "L.2 findings").
SINGLES = ("c172", "sr22", "sf50")  # one engine (the SR22 and SF50: two generators)
NO_TAXI_OR_BEACON = ("sr22", "sf50")
NO_SIGNS = ("baron", "c172", "sr22", "sf50")
NO_YAW_DAMPER = ("a330", "c172", "sr22")
NO_AVIONICS = ("737", "a330", "md82", "sf50")


def _all(*checks: SimCheck) -> tuple[SimCheck, ...]:
    return checks


def _is(what: str, dataref: str, want: float, tol: float = 0.0) -> SimCheck:
    return equals(what, dataref, want, tol=tol, within_sec=5)


def _between(what: str, dataref: str, low: int, high: int) -> SimCheck:
    return SimCheck(
        what,
        (dataref,),
        lambda v: v is not None and low <= round(float(v)) <= high,
        within_sec=5,
        expected=f"{low}-{high}",
    )


def _set(name: str, say: str, *checks: SimCheck, note: str = "") -> Step:
    return Step(name, say=say, outcomes=SET_OR_TURNING, sim=checks, note=note)


def _refused(name: str, say: str, reply: str, *checks: SimCheck) -> Step:
    return Step(name, say=say, outcomes=("rejected",), reply=reply, vimaan_checks=False, sim=checks)


def _pitot_off_held(*still_on: SimCheck, no_off: bool = False) -> list[Step]:
    """Pitot heat off in the air is held, then cancelled. Where the aircraft
    has no off (the A330), the repeat confirms it and Vimaan asks "on or
    auto?", so nothing changes either way."""
    held = Step(
        "probe heat off is held in flight",
        say=PITOT_OFF_SAY,
        outcomes=("safety_hold",),
        reply="Confirm pitot heat off",
        vimaan_checks=False,
        sim=still_on,
    )
    if no_off:
        then = _refused("confirmed, probe heat has no off", PITOT_OFF_SAY, "auto", *still_on)
    else:
        then = Step(
            "cancel the probe heat off",
            say="cancel",
            outcomes=("cancelled",),
            vimaan_checks=False,
            sim=still_on,
        )
    return [held, dataclasses.replace(then, after=held.name)]


def _lights(aircraft: str) -> list[Step]:
    one_light = [_is("landing light", SW + "landing_lights_on", 1)]
    landing = {
        "737": [_is(f"landing light {i}", f"{SW}landing_lights_switch[{i}]", 1) for i in range(4)],
        "a330": [_is("LAND", f"{SW}landing_lights_switch[0]", 1)],
        "c172": one_light,
        "sr22": one_light,
        "sf50": one_light,
        "md82": [
            _is(f"landing light {s}", f"{SW}landing_lights_switch[{i}]", 1)
            for s, i in (("L", 1), ("R", 2))
        ],
    }.get(
        aircraft,
        [_is(f"landing light {i + 1}", f"{SW}landing_lights_switch[{i}]", 1) for i in (0, 1)],
    )
    strobes = {
        "737": _is("position switch STROBE & STEADY", B738 + "position_light_pos", 1),
        "a330": _is("strobe ON", "laminar/a333/switches/strobe_pos", 2),
    }.get(aircraft, _is("strobes", SW + "strobe_lights_on", 1))
    nav = {
        "a330": _between("nav & logo on", "laminar/a333/switches/nav_pos", 1, 2),
    }.get(aircraft, _is("nav lights", SW + "navigation_lights_on", 1))
    taxi_off = {
        "737": _is("taxi switch off", B738 + "taxi_light_brightness_pos", 0),
        "a330": _is("nose light off", f"{SW}landing_lights_switch[1]", 0),
    }.get(aircraft, _is("taxi light off", SW + "taxi_light_on", 0))
    if aircraft in NO_TAXI_OR_BEACON:
        beacon = _refused("no beacon", "beacon on", "no beacon")
        taxi = _refused("no taxi lights", "taxi lights off", "no taxi")
    else:
        beacon = _set("beacon on", "beacon on", _is("beacon", SW + "beacon_on", 1))
        taxi = _set("taxi lights off", "taxi lights off", taxi_off)
    out = [
        beacon,
        _set("landing lights on", "landing lights on", *landing),
        _set("strobes on", "strobes on", strobes),
        _set("nav lights on", "nav lights on", nav),
        taxi,
    ]
    if aircraft == "a330":
        out.append(
            _set(
                "strobes auto",
                "strobes auto",
                _is("strobe AUTO", "laminar/a333/switches/strobe_pos", 1),
            )
        )
    if aircraft == "737":
        out.append(
            _set(
                "logo lights on", "logo lights on", _is("logo", f"{SW}generic_lights_switch[1]", 1)
            )
        )
    elif aircraft == "a330":
        out.append(_set("logo lights on", "logo lights on", nav))
    elif aircraft in ("citx", "md82"):  # the tail flood lights
        logo = _is("logo", f"{SW}generic_lights_switch[0]", 1)
        out.append(_set("logo lights on", "logo lights on", logo))
    else:
        out.append(_refused("no logo lights", "logo lights on", "no logo"))
    return out


def _signs(aircraft: str) -> list[Step]:
    if aircraft in NO_SIGNS:
        return [_refused("no seat belt sign", "seat belt sign on", "no seat belt")]
    if aircraft == "citx":  # one switch: SEAT BELT / OFF / PASS SAFETY (both)
        pos = CITX + "lights/seat_belt_pass_safety_term"
        return [
            _set(
                "seat belt sign on",
                "seat belt sign on",
                _is("switch at SEAT BELT", pos, -1),
                _is("seat belt sign", SW + "fasten_seat_belts", 1),
            ),
            _refused("seat belt sign has no auto", "seat belt sign auto", "on or off"),
            _set(
                "no smoking sign on",
                "no smoking sign on",
                _is("switch at PASS SAFETY", pos, 1),
                _is("no smoking sign", SW + "no_smoking", 1),
            ),
        ]
    if aircraft == "md82":  # X-Plane's sign datarefs: 0 OFF, 1 AUTO, 2 ON
        return [
            _set(
                "seat belt sign on",
                "seat belt sign on",
                _is("seat belts ON", SW + "fasten_seat_belts", 2),
            ),
            _set(
                "seat belt sign auto",
                "seat belt sign auto",
                _is("seat belts AUTO", SW + "fasten_seat_belts", 1),
            ),
            _set(
                "no smoking sign auto",
                "no smoking sign auto",
                _is("no smoking AUTO", SW + "no_smoking", 1),
            ),
        ]
    if aircraft == "kingair":
        pos = C90 + "lighting/switch/no_smoke_fsb_pos"
        return [
            _set(
                "seat belt sign on",
                "seat belt sign on",
                _is("seat belt sign", SW + "fasten_seat_belts", 1),
            ),
            _refused("seat belt sign has no auto", "seat belt sign auto", "on or off"),
            _set("no smoking sign on", "no smoking sign on", _is("switch at NO SMOKING", pos, 1)),
        ]
    if aircraft == "737":
        return [
            _set(
                "seat belt sign on",
                "seat belt sign on",
                _is("seat belt switch ON", B738 + "seatbelt_sign_pos", 2),
            ),
            _set(
                "seat belt sign auto",
                "seat belt sign auto",
                _is("seat belt switch AUTO", B738 + "seatbelt_sign_pos", 1),
            ),
            _refused("no no smoking switch", "no smoking sign on", "no no smoking"),
        ]
    return [
        _set(
            "seat belt sign on",
            "seat belt sign on",
            _is("seat belts ON", A333 + "switches/fasten_seatbelts", 2),
        ),
        _set(
            "seat belt sign auto",
            "seat belt sign auto",
            _is("seat belts AUTO", A333 + "switches/fasten_seatbelts", 1),
        ),
        _set(
            "no smoking sign auto",
            "no smoking sign auto",
            _is("no smoking AUTO", A333 + "switches/no_smoking", 1),
        ),
    ]


def _anti_ice(aircraft: str) -> list[Step]:
    if aircraft == "a330":
        heat = A333 + "monitors/probe_window_heat"
        return [
            _set("probe heat on", "probe heat on", _is("probe / window heat ON", heat, 1)),
            *_pitot_off_held(_is("still ON", heat, 1), no_off=True),
            _set("probe heat auto", "probe heat auto", _is("probe / window heat AUTO", heat, 0)),
            _set(
                "wing anti ice on",
                "wing anti ice on",
                _is("wing anti-ice", A333 + "buttons/wing_anti_ice", 1),
            ),
            _set(
                "wing anti ice off",
                "wing anti ice off",
                _is("wing anti-ice off", A333 + "buttons/wing_anti_ice", 0),
            ),
            _set(
                "engine anti ice on",
                "engine anti ice on",
                *(
                    _is(f"engine {n} anti-ice", f"{A333}buttons/engine_anti_ice{n}", 1)
                    for n in (1, 2)
                ),
            ),
            _set(
                "engine anti ice off",
                "engine anti ice off",
                *(
                    _is(f"engine {n} anti-ice off", f"{A333}buttons/engine_anti_ice{n}", 0)
                    for n in (1, 2)
                ),
            ),
            _refused("no prop heat", "prop heat on", "no prop"),
        ]
    if aircraft == "737":
        probes = (B738 + "capt_probes_pos", B738 + "fo_probes_pos")
        return [
            _set(
                "probe heat on",
                "probe heat on",
                *(_is(f"probe heat {ab}", d, 1) for ab, d in zip("AB", probes)),
            ),
            *_pitot_off_held(
                *(_is(f"probe heat {ab} still on", d, 1) for ab, d in zip("AB", probes))
            ),
            _set(
                "window heat on",
                "window heat on",
                *(
                    _is(f"window {i + 1}", f"{ICE}ice_window_heat_on_window[{i}]", 1)
                    for i in range(4)
                ),
            ),
            _set(
                "wing anti ice on",
                "wing anti ice on",
                _is("wing anti-ice", ICE + "ice_surfce_heat_on", 1),
            ),
            _set(
                "wing anti ice off",
                "wing anti ice off",
                _is("wing anti-ice off", ICE + "ice_surfce_heat_on", 0),
            ),
            _set(
                "engine anti ice on",
                "engine anti ice on",
                *(
                    _is(f"engine {i + 1} anti-ice", f"{ICE}ice_inlet_heat_on_per_engine[{i}]", 1)
                    for i in (0, 1)
                ),
            ),
            _set(
                "engine anti ice off",
                "engine anti ice off",
                *(
                    _is(
                        f"engine {i + 1} anti-ice off", f"{ICE}ice_inlet_heat_on_per_engine[{i}]", 0
                    )
                    for i in (0, 1)
                ),
            ),
            _refused("no prop heat", "prop heat on", "no prop"),
        ]
    if aircraft in ("citx", "md82"):
        return _jet_anti_ice(aircraft)
    if aircraft in SINGLES:
        return _single_anti_ice(aircraft)
    pitot = (
        _is("pitot heat left", ICE + "ice_pitot_heat_on_pilot", 1),
        _is("pitot heat right", ICE + "ice_pitot_heat_on_copilot", 1),
    )
    if aircraft == "kingair":
        windshield = (
            C90 + "iceRain/switch/windshield_antiice_pos_L",
            C90 + "iceRain/switch/windshield_antiice_pos_R",
        )
        return [
            _set("pitot heat on", PITOT_SAY, *pitot),
            *_pitot_off_held(*pitot),
            _set(
                "window heat on",
                "window heat on",
                *(_is(f"windshield {s} NORMAL", d, 1) for s, d in zip("LR", windshield)),
            ),
            _set(
                "boots on",
                "boots on",
                _is("boots cycling", ICE + "ice_surface_boot_on", 3),
                note="one 7-second cycle",
            ),
            _refused("boots have no off", "boots off", "cycle once"),
            _set(
                "engine anti ice on",
                "engine anti ice on",
                *(_is(f"ice vane {i + 1}", f"{ICE}anti_ice_engine_air[{i}]", 1) for i in (0, 1)),
            ),
            _set(
                "engine anti ice off",
                "engine anti ice off",
                *(
                    _is(f"ice vane {i + 1} off", f"{ICE}anti_ice_engine_air[{i}]", 0)
                    for i in (0, 1)
                ),
            ),
            _set(
                "prop heat on",
                PROP_SAY["on"],
                _is("prop de-ice switch", C90 + "iceRain/switch/prop_deice_pos", 1),
            ),
            _set(
                "prop heat off",
                PROP_SAY["off"],
                _is("prop de-ice off", C90 + "iceRain/switch/prop_deice_pos", 0),
            ),
        ]
    return [  # Baron
        _set("pitot heat on", PITOT_SAY, *pitot),
        *_pitot_off_held(*pitot),
        _set("window heat on", "window heat on", _is("window heat", ICE + "ice_window_heat_on", 1)),
        _set("boots on", "boots on", _is("boots AUTO", ICE + "ice_surface_boot_on", 2)),
        _set("boots off", "boots off", _is("boots off", ICE + "ice_surface_boot_on", 0)),
        _refused("no engine anti ice", "engine anti ice on", "no engine anti"),
        _set("prop heat on", PROP_SAY["on"], _is("prop heat", ICE + "ice_prop_heat_on", 1)),
        _set("prop heat off", PROP_SAY["off"], _is("prop heat off", ICE + "ice_prop_heat_on", 0)),
    ]


def _on_off_pair(name: str, say: str, *checks: tuple[str, str]) -> list[Step]:
    """ "<say> on", then "<say> off", each checked on every (what, dataref)."""
    return [
        _set(
            f"{name} {state}", f"{say} {state}", *(_is(f"{w} {state}", d, want) for w, d in checks)
        )
        for state, want in (("on", 1), ("off", 0))
    ]


def _jet_anti_ice(aircraft: str) -> list[Step]:
    """The Citation X and MD-82: pitot L / R, windshield, wing, two engines."""
    pitot = (
        _is("pitot heat left", ICE + "ice_pitot_heat_on_pilot", 1),
        _is("pitot heat right", ICE + "ice_pitot_heat_on_copilot", 1),
    )
    if aircraft == "citx":
        window = tuple(
            _is(f"windshield {s} ON", f"{CITX}ice/windshield_{side}_term", 1)
            for s, side in (("L", "left"), ("R", "right"))
        )
        wing = (("slat anti-ice", ICE + "ice_surface_hot_bleed_air_on"),)
        engines = tuple(
            (f"engine {i + 1} anti-ice", f"{ICE}cowling_thermal_anti_ice_per_engine[{i}]")
            for i in (0, 1)
        )
    else:
        window = (_is("windshield heat", ICE + "ice_window_heat_on", 1),)
        wing = tuple(
            (f"wing heat {s}", f"{ICE}ice_surfce_heat_{side}_on")
            for s, side in (("L", "left"), ("R", "right"))
        )
        engines = tuple(
            (f"engine {i + 1} anti-ice", f"{ICE}ice_inlet_heat_on_per_engine[{i}]") for i in (0, 1)
        )
    return [
        _set("probe heat on", "probe heat on", *pitot),
        *_pitot_off_held(*pitot),
        _set("window heat on", "window heat on", *window),
        *_on_off_pair("wing anti ice", "wing anti ice", *wing),
        *_on_off_pair("engine anti ice", "engine anti ice", *engines),
        _refused("no prop heat", "prop heat on", "no prop"),
    ]


def _single_anti_ice(aircraft: str) -> list[Step]:
    """The C172, SR22 and SF50: one pitot heat (the pilot's); the C172 has
    nothing else, the SR22 window heat and its TKS fluid (the wing anti-ice),
    the SF50 window, wing and engine inlet heat."""
    pitot = (_is("pitot heat", ICE + "ice_pitot_heat_on_pilot", 1),)
    out = [_set("pitot heat on", PITOT_SAY, *pitot), *_pitot_off_held(*pitot)]
    if aircraft == "c172":
        return out + [
            _refused("no window heat", "window heat on", "no window heat"),
            _refused("no wing anti ice", "wing anti ice on", "no wing anti"),
            _refused("no engine anti ice", "engine anti ice on", "no engine anti"),
            _refused("no prop heat", "prop heat on", "no prop"),
        ]
    out.append(
        _set("window heat on", "window heat on", _is("window heat", ICE + "ice_window_heat_on", 1))
    )
    if aircraft == "sr22":
        return out + [
            *_on_off_pair(
                "wing anti ice (TKS)",
                "wing anti ice",
                ("ice protection", "laminar/sr22/switch/ice_protect"),
            ),
            _refused("no engine anti ice", "engine anti ice on", "no engine anti"),
            _refused("no prop heat", "prop heat on", "no prop"),
        ]
    return out + [  # SF50
        *_on_off_pair("wing anti ice", "wing anti ice", ("wing heat", ICE + "ice_surfce_heat_on")),
        *_on_off_pair(
            "engine anti ice",
            "engine anti ice",
            ("engine inlet heat", ICE + "ice_inlet_heat_on_per_engine[0]"),
        ),
        _refused("no prop heat", "prop heat on", "no prop"),
    ]


def _power_and_fuel(aircraft: str) -> list[Step]:
    out = []
    if aircraft in NO_AVIONICS:
        out.append(_refused("no avionics switch", "avionics on", "no avionics"))
    elif aircraft == "citx":
        out.append(
            _set(
                "avionics on", "avionics on", _is("avionics", CITX + "electrical/avionics_term", 1)
            )
        )
    else:
        out.append(_set("avionics on", "avionics on", _is("avionics", SW + "avionics_power_on", 1)))
    gen2 = {
        "a330": A333 + "buttons/gen2_pos",
        "kingair": C90 + "electrical/switch/gen2_switch_pos",
        "citx": CITX + "electrical/generator_right_term",
        "c172": EL + "generator_on[0]",  # its one alternator
    }.get(aircraft, EL + "generator_on[1]")
    gen2_on = 0 if aircraft == "kingair" else 1  # the King Air's ON is the middle position
    out += [
        Step(
            "generator off is held in flight",
            say="generator off" if aircraft == "c172" else "generator two off",
            outcomes=("safety_hold",),
            reply="Confirm power off",
            vimaan_checks=False,
            sim=(_is("generator 2 still on", gen2, gen2_on),),
        ),
        Step(
            "cancel the generator off",
            say="cancel",
            outcomes=("cancelled",),
            vimaan_checks=False,
            after="generator off is held in flight",
            sim=(equals("generator 2 still on", gen2, gen2_on, within_sec=2),),
        ),
    ]
    if aircraft == "737":
        pumps = [
            _is(f"tank pumps {t}", f"sim/cockpit2/fuel/fuel_tank_pump_on[{i}]", 1)
            for i, t in enumerate(("left", "center", "right"))
        ]
    elif aircraft == "a330":
        pumps = [
            _is(f"pump {n}", f"{A333}fuel/buttons/{n}_pump_pos", 1)
            for n in ("left1", "left2", "right1", "right2", "center_left", "center_right")
        ]
    elif aircraft == "md82":  # tank pumps: [0] center, [1] left, [2] right
        pumps = [
            _is(f"tank pumps {t}", f"sim/cockpit2/fuel/fuel_tank_pump_on[{i}]", 1)
            for i, t in ((1, "left"), (0, "center"), (2, "right"))
        ]
    elif aircraft == "citx":
        pumps = [
            _is(f"boost pump {s} ON", f"{CITX}fuel/boost_{side}_term", 1)
            for s, side in (("L", "left"), ("R", "right"))
        ]
    elif aircraft == "sr22":
        pumps = [_is("boost pump BOOST", "laminar/sr22/switch/boost_pump", 1)]
    elif aircraft == "sf50":  # its pump follows the ignition knob
        return out + [_refused("no fuel pump switch", "fuel pumps on", "no fuel pump")]
    elif aircraft == "c172":
        pumps = [_is("fuel pump", "sim/cockpit2/engine/actuators/fuel_pump_on[0]", 1)]
    else:
        pumps = [
            _is(f"boost pump {i + 1}", f"sim/cockpit2/engine/actuators/fuel_pump_on[{i}]", 1)
            for i in (0, 1)
        ]
    out += [
        _set("fuel pumps on", "fuel pumps on", *pumps),
        Step(
            "fuel pumps off is held in flight",
            say="fuel pumps off",
            outcomes=("safety_hold",),
            reply="Confirm fuel pump off",
            vimaan_checks=False,
            sim=(pumps[0],),
        ),
        Step(
            "cancel the fuel pumps off",
            say="cancel",
            outcomes=("cancelled",),
            vimaan_checks=False,
            after="fuel pumps off is held in flight",
            sim=(pumps[0],),
        ),
    ]
    return out


def _jets(aircraft: str) -> list[Step]:
    if aircraft in ("citx", "md82"):
        ratio = "sim/cockpit2/controls/speedbrake_ratio"
        out = [
            _set(
                "speed brake armed",
                "speed brake armed",
                _is("speed brake armed", ratio, -0.5, tol=0.01),
            ),
            _set(
                "speed brake retract", "stow the speed brakes", _is("retracted", ratio, 0, tol=0.01)
            ),
        ]
        if aircraft == "citx":
            return out + [_refused("no auto brake", "auto brake two", "no auto brake")]
        return out + [
            _set(
                "auto brake two", "auto brake two", _is("autobrake 2", SW + "auto_brake_level", 3)
            ),
            _set(
                "auto brake off", "auto brake off", _is("autobrake off", SW + "auto_brake_level", 1)
            ),
        ]
    if aircraft not in ("737", "a330"):
        return [
            _refused("no speed brakes", "speed brake armed", "no speed brakes"),
            _refused("no auto brake", "auto brake two", "no auto brake"),
        ]
    if aircraft == "737":
        lever = "laminar/B738/flt_ctrls/speedbrake_lever"
        brakes = [
            _set(
                "speed brake armed",
                "speed brake armed",
                _is("lever ARMED", lever, 0.0889, tol=0.01),
            ),
            _set(
                "speed brake retract",
                "stow the speed brakes",
                _is("lever DOWN", lever, 0, tol=0.01),
            ),
            _set(
                "auto brake two", "auto brake two", _is("autobrake 2", SW + "auto_brake_level", 3)
            ),
        ]
    else:
        ratio = "sim/cockpit2/controls/speedbrake_ratio"
        brakes = [
            _set(
                "speed brake armed",
                "speed brake armed",
                _is("ground spoilers armed", ratio, -0.5, tol=0.01),
            ),
            _set(
                "speed brake retract", "stow the speed brakes", _is("retracted", ratio, 0, tol=0.01)
            ),
            _set(
                "auto brake medium",
                "auto brake medium",
                _is("autobrake MED", SW + "auto_brake_level", 4),
                note="needs nosewheel steering on",
            ),
        ]
    return [
        *brakes,
        _set("auto brake off", "auto brake off", _is("autobrake off", SW + "auto_brake_level", 1)),
    ]


def steps(aircraft: str) -> list[Step]:
    if aircraft not in SETUPS:
        raise KeyError(aircraft)
    out = hands_off(aircraft)
    out += _lights(aircraft)
    out += _signs(aircraft)
    if aircraft in NO_YAW_DAMPER:
        out.append(_refused("no yaw damper switch", "yaw damper on", "no yaw damper"))
    else:
        out.append(
            _set("yaw damper on", "yaw damper on", _is("yaw damper", SW + "yaw_damper_on", 1))
        )
    out += _anti_ice(aircraft)
    out += _power_and_fuel(aircraft)
    out += _jets(aircraft)
    return out
