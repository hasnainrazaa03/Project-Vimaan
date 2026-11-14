"""Cockpit switches (Tier 3): how each setting is reached, as steps.

A *plan* is a tuple of steps the plugin runs one per frame, each waiting for
the aircraft to show its result before the next:

- ``Knob``: a selector clicked toward a position range, one click at a time
  (Tier 2's transponder selectors work the same way).
- ``Button``: press ``command`` unless ``state`` already reads ``want``, then
  wait for it to arrive. A generic on/off command, or a toggle pushbutton
  whose state can be read (the A330's).
- ``Write``: write a dataref and check that it holds (the 737's light
  switches and speed brake lever are plain datarefs).
- ``Check``: only as a plan's first step. When ``state`` already reads
  ``want`` the setting is already made and nothing is pressed. The 737's
  strobes share one switch with its nav lights, and X-Plane's own light
  datarefs say which are on.

``GENERIC`` holds the plans for any aircraft. An aircraft's override replaces
entries (``Override.settings["switches"]``). A key mapped to ``None`` means
the aircraft has no such switch; a setting missing from a key's plans means
that switch has no such position. The findings behind every entry are in
docs/AIRCRAFT_PROFILES.md, "Tier 3 findings" (X-Plane 12.4.3).

Pure Python: no X-Plane import, so it can be tested anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Knob:
    """A cockpit selector turned one click at a time, on aircraft whose
    scripts own a setting and put a written value back (the 737's and
    A330's transponder mode). ``position`` is the aircraft's dataref for
    where it sits; ``down`` / ``up`` turn it one click toward a lower /
    higher position. Done when the position is in ``low``..``high``."""

    position: str
    down: str
    up: str
    low: int
    high: int

    def at(self, low: int, high: int | None = None) -> Knob:
        return Knob(self.position, self.down, self.up, low, low if high is None else high)


@dataclass(frozen=True)
class Button:
    """Press ``command`` unless ``state`` already reads one of ``want``."""

    state: str
    command: str
    want: tuple[float, ...]

    def __init__(self, state: str, command: str, want: float | tuple[float, ...]):
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "want", want if isinstance(want, tuple) else (float(want),))


@dataclass(frozen=True)
class Write:
    """Write ``value`` to ``dataref`` ("name[i]" for an array element) and
    check that it holds."""

    dataref: str
    value: float


@dataclass(frozen=True)
class Check:
    """First step only: the setting is already made when ``state`` reads ``want``."""

    state: str
    want: float


Step = Knob | Button | Write | Check
Plan = tuple[Step, ...]
# key ("light:landing") -> setting ("on") -> plan; None: no such switch.
Plans = Mapping[str, Mapping[str, Plan] | None]

TOLERANCE = 0.02  # a switch value read back within this counts as set


def reads(value: float | None, want: tuple[float, ...]) -> bool:
    return value is not None and any(abs(value - w) <= TOLERANCE for w in want)


def names(step: Step) -> tuple[str, ...]:
    """Every dataref and command a step uses ("name[i]" with its index)."""
    if isinstance(step, Knob):
        return (step.position, step.down, step.up)
    if isinstance(step, Button):
        return (step.state, step.command)
    if isinstance(step, Write):
        return (step.dataref,)
    return (step.state,)


def plan_names(plans: Plans) -> set[str]:
    out: set[str] = set()
    for table in plans.values():
        for plan in (table or {}).values():
            for step in plan:
                out.update(names(step))
    return out


def _frozen(d: Mapping) -> Mapping:
    return MappingProxyType(dict(d))


SW = "sim/cockpit2/switches/"
SPEEDBRAKE = "sim/cockpit2/controls/speedbrake_ratio"
AUTOBRAKE = SW + "auto_brake_level"
# Says whether the aircraft has speed brakes at all (0 on the Baron and King Air).
HAS_SPEEDBRAKE = "sim/aircraft/parts/acf_sbrkEQ"


def _on_off(state: str, on: str, off: str) -> Mapping[str, Plan]:
    return _frozen({"on": (Button(state, on, 1),), "off": (Button(state, off, 0),)})


def _toggled(state: str, toggle: str) -> Mapping[str, Plan]:
    return _frozen({"on": (Button(state, toggle, 1),), "off": (Button(state, toggle, 0),)})


def _written(*datarefs: str) -> Mapping[str, Plan]:
    return _frozen(
        {
            "on": tuple(Write(d, 1) for d in datarefs),
            "off": tuple(Write(d, 0) for d in datarefs),
        }
    )


def _lights(name: str) -> tuple[str, str]:
    return f"sim/lights/{name}_on", f"sim/lights/{name}_off"


# The settings each key can be asked for (speed brake and autobrake values
# are their slots' canonical values).
LIGHTS = ("landing", "taxi", "beacon", "strobe", "nav", "logo", "wing", "runway")
SIGNS = ("seat belt", "no smoking")
SPEEDBRAKE_SETTINGS = ("retract", "armed", "half", "extend")
AUTOBRAKE_SETTINGS = ("off", "rto", "1", "2", "3", "max")

GENERIC: Plans = _frozen(
    {
        "light:landing": _on_off(SW + "landing_lights_on", *_lights("landing_lights")),
        "light:taxi": _on_off(SW + "taxi_light_on", *_lights("taxi_lights")),
        "light:beacon": _on_off(SW + "beacon_on", *_lights("beacon_lights")),
        "light:strobe": _on_off(SW + "strobe_lights_on", *_lights("strobe_lights")),
        "light:nav": _on_off(SW + "navigation_lights_on", *_lights("nav_lights")),
        # Which generic light is the logo, wing or runway turnoff light differs
        # per aircraft; only known aircraft have them.
        "light:logo": None,
        "light:wing": None,
        "light:runway": None,
        "sign:seat belt": _toggled(SW + "fasten_seat_belts", "sim/systems/seatbelt_sign_toggle"),
        "sign:no smoking": _toggled(SW + "no_smoking", "sim/systems/no_smoking_toggle"),
        "yaw_damper": _on_off(
            SW + "yaw_damper_on", "sim/systems/yaw_damper_on", "sim/systems/yaw_damper_off"
        ),
        # -0.5 armed, 0 retracted, 1 fully extended (X-Plane's handle ratio).
        "speed_brake": _frozen(
            {
                "retract": (Write(SPEEDBRAKE, 0.0),),
                "armed": (Write(SPEEDBRAKE, -0.5),),
                "half": (Write(SPEEDBRAKE, 0.5),),
                "extend": (Write(SPEEDBRAKE, 1.0),),
            }
        ),
        # 0 RTO, 1 off, 2-5 increasing levels ("max" is 5).
        "autobrake": _frozen(
            {
                setting: (Write(AUTOBRAKE, level),)
                for setting, level in (
                    ("rto", 0),
                    ("off", 1),
                    ("1", 2),
                    ("2", 3),
                    ("3", 4),
                    ("max", 5),
                )
            }
        ),
    }
)

# --- Beechcraft Baron 58 ---------------------------------------------------
# Every switch clicks a generic command and animates on the generic dataref.
# The two landing lights are separate switches (landing_lights_on is the
# first only); the "wing" light is its wing-ice light. No cabin signs.
B58: Plans = _frozen(
    {
        "light:landing": _frozen(
            {
                "on": (
                    Button(SW + "landing_lights_switch[0]", "sim/lights/landing_01_light_on", 1),
                    Button(SW + "landing_lights_switch[1]", "sim/lights/landing_02_light_on", 1),
                ),
                "off": (
                    Button(SW + "landing_lights_switch[0]", "sim/lights/landing_01_light_off", 0),
                    Button(SW + "landing_lights_switch[1]", "sim/lights/landing_02_light_off", 0),
                ),
            }
        ),
        "light:wing": _written(SW + "generic_lights_switch[0]"),
        "sign:seat belt": None,
        "sign:no smoking": None,
    }
)

# --- Beechcraft King Air C90B ----------------------------------------------
# One three-position sign switch: up NO SMOKING + SEAT BELT (1), OFF (0),
# down SEAT BELT only (-1). Its position dataref is read-only (xlua
# create_dataref without a handler), so it's clicked with its commands.
# Its handler skips the sign when going up from -1 to 0, leaving the seat
# belt sign lit: the toggle afterwards puts it out.
C90_SIGNS = Knob(
    "laminar/c90/lighting/switch/no_smoke_fsb_pos",
    "laminar/c90/lighting/switch/no_smoke_fsb_dn",
    "laminar/c90/lighting/switch/no_smoke_fsb_up",
    -1,
    1,
)
C90: Plans = _frozen(
    {
        "light:landing": B58["light:landing"],
        "light:wing": _on_off(
            SW + "generic_lights_switch[1]",
            "laminar/c90/lighting/switch/wing_ice_up",
            "laminar/c90/lighting/switch/wing_ice_dn",
        ),
        "sign:seat belt": _frozen(
            {
                "on": (Check(SW + "fasten_seat_belts", 1), C90_SIGNS.at(-1)),
                "off": (
                    Check(SW + "fasten_seat_belts", 0),
                    C90_SIGNS.at(0),
                    Button(SW + "fasten_seat_belts", "sim/systems/seatbelt_sign_toggle", 0),
                ),
            }
        ),
        "sign:no smoking": _frozen(
            {
                "on": (Check(SW + "no_smoking", 1), C90_SIGNS.at(1)),
                # No smoking alone isn't a position: off leaves the seat belt sign.
                "off": (Check(SW + "no_smoking", 0), C90_SIGNS.at(-1)),
            }
        ),
    }
)

# --- Boeing 737-800 ----------------------------------------------------------
# Its own switch datarefs are read-only, except the speed brake lever.
B738_TAXI = Knob(  # 0 off, 1, 2 (full); its "up" command turns it down
    "laminar/B738/toggle_switch/taxi_light_brightness_pos",
    "laminar/B738/toggle_switch/taxi_light_brightness_pos_up",
    "laminar/B738/toggle_switch/taxi_light_brightness_pos_dn",
    0,
    2,
)
B738_POSITION = Knob(  # -1 STEADY (nav), 0 OFF, 1 STROBE & STEADY
    "laminar/B738/toggle_switch/position_light_pos",
    "laminar/B738/toggle_switch/position_light_down",
    "laminar/B738/toggle_switch/position_light_up",
    -1,
    1,
)
B738_SEATBELT = Knob(  # 0 OFF, 1 AUTO, 2 ON; its "up" command turns it toward OFF
    "laminar/B738/toggle_switch/seatbelt_sign_pos",
    "laminar/B738/toggle_switch/seatbelt_sign_up",
    "laminar/B738/toggle_switch/seatbelt_sign_dn",
    0,
    2,
)
# DOWN 0, ARMED 0.0889, FLIGHT DETENT 0.667, UP 1.0 (in the air its script
# holds the lever at the flight detent or below). Disarming through X-Plane's
# ratio leaves the lever showing ARMED, so the lever itself is written.
B738_SPEEDBRAKE = "laminar/B738/flt_ctrls/speedbrake_lever"
B738: Plans = _frozen(
    {
        # [1], [2] the retractable belly lights, [0], [3] the fixed wing-root ones.
        "light:landing": _written(*(f"{SW}landing_lights_switch[{i}]" for i in range(4))),
        "light:taxi": _frozen({"on": (B738_TAXI.at(2),), "off": (B738_TAXI.at(0),)}),
        "light:strobe": _frozen(
            {
                "on": (Check(SW + "strobe_lights_on", 1), B738_POSITION.at(1)),
                "off": (Check(SW + "strobe_lights_on", 0), B738_POSITION.at(-1)),
            }
        ),
        "light:nav": _frozen(
            {
                "on": (Check(SW + "navigation_lights_on", 1), B738_POSITION.at(-1)),
                "off": (Check(SW + "navigation_lights_on", 0), B738_POSITION.at(0)),
            }
        ),
        "light:logo": _written(SW + "generic_lights_switch[1]"),
        "light:wing": _written(SW + "generic_lights_switch[0]"),
        "light:runway": _written(SW + "generic_lights_switch[2]", SW + "generic_lights_switch[3]"),
        "sign:seat belt": _frozen(
            {
                "on": (B738_SEATBELT.at(2),),
                "auto": (B738_SEATBELT.at(1),),
                "off": (B738_SEATBELT.at(0),),
            }
        ),
        "sign:no smoking": None,  # no switch modelled
        "speed_brake": _frozen(
            {
                "retract": (Write(B738_SPEEDBRAKE, 0.0),),
                "armed": (Write(B738_SPEEDBRAKE, 0.0889),),
                "half": (Write(B738_SPEEDBRAKE, 0.667),),
                "extend": (Write(B738_SPEEDBRAKE, 1.0),),
            }
        ),
    }
)

# --- Airbus A330-300 ---------------------------------------------------------
# Its scripts rewrite the generic strobe, nav and sign datarefs every frame
# from its own switches, whose datarefs are read-only: the switches are moved
# with its commands. Its pushbuttons toggle.
A333_STROBE = "laminar/a333/switches/strobe_pos"  # 0 OFF, 1 AUTO, 2 ON
A333_NAV = Knob(  # nav & logo: 0 OFF, 1, 2
    "laminar/a333/switches/nav_pos",
    "laminar/A333/toggle_switch/nav_light_pos_dn",
    "laminar/A333/toggle_switch/nav_light_pos_up",
    0,
    2,
)
A333_SEATBELT = Knob(  # 0 OFF, 1 AUTO, 2 ON
    "laminar/A333/switches/fasten_seatbelts",
    "laminar/A333/switches/seatbelt_signs_dn",
    "laminar/A333/switches/seatbelt_signs_up",
    0,
    2,
)
A333_NO_SMOKING = Knob(  # 0 OFF, 1 AUTO, 2 ON
    "laminar/A333/switches/no_smoking",
    "laminar/A333/switches/smoking_signs_dn",
    "laminar/A333/switches/smoking_signs_up",
    0,
    2,
)
A333_NOSE = SW + "landing_lights_switch[1]"  # nose light: 0 OFF, 0.5 TAXI, 1 T.O.


def _three(knob: Knob) -> Mapping[str, Plan]:
    return _frozen({"on": (knob.at(2),), "auto": (knob.at(1),), "off": (knob.at(0),)})


A333: Plans = _frozen(
    {
        # One LAND switch, on the first landing light.
        "light:landing": _on_off(SW + "landing_lights_switch[0]", *_lights("landing_01_light")),
        # No taxi light: "taxi lights" is the nose light at TAXI.
        "light:taxi": _frozen(
            {
                "on": (Button(A333_NOSE, "laminar/A333/switch/lighting/taxi_light_taxi", 0.5),),
                "off": (Button(A333_NOSE, "laminar/A333/switch/lighting/taxi_light_off", 0),),
            }
        ),
        "light:strobe": _frozen(
            {
                "on": (Button(A333_STROBE, "laminar/A333/toggle_switch/strobe_pos_on", 2),),
                "auto": (Button(A333_STROBE, "laminar/A333/toggle_switch/strobe_pos_auto", 1),),
                "off": (Button(A333_STROBE, "laminar/A333/toggle_switch/strobe_pos_off", 0),),
            }
        ),
        "light:nav": _frozen({"on": (A333_NAV.at(1, 2),), "off": (A333_NAV.at(0),)}),
        "light:logo": _frozen({"on": (A333_NAV.at(1, 2),), "off": (A333_NAV.at(0),)}),
        "light:wing": _written(SW + "generic_lights_switch[1]"),
        "light:runway": _on_off(
            SW + "generic_lights_switch[0]",
            "laminar/A333/toggle_switch/rwy_turn_off_on",
            "laminar/A333/toggle_switch/rwy_turn_off_off",
        ),
        "sign:seat belt": _three(A333_SEATBELT),
        "sign:no smoking": _three(A333_NO_SMOKING),
        "yaw_damper": None,  # fly-by-wire: no pilot switch
        # LO / MED / MAX pushbuttons (X-Plane levels 3 / 4 / RTO or 5); each
        # toggles, so it's pressed only when its level isn't already set.
        # Off: the lit one is pressed (A333_AUTOBRAKE_OFF). No "3".
        "autobrake": _frozen(
            {
                "1": (Button(AUTOBRAKE, "sim/flight_controls/brakes_1_auto", 3),),
                "2": (Button(AUTOBRAKE, "sim/flight_controls/brakes_2_auto", 4),),
                "max": (Button(AUTOBRAKE, "sim/flight_controls/brakes_rto_auto", (0, 5)),),
                "rto": (Button(AUTOBRAKE, "sim/flight_controls/brakes_rto_auto", (0, 5)),),
            }
        ),
    }
)
# Autobrake level -> the A330 pushbutton that is lit at that level.
A333_AUTOBRAKE_OFF: Mapping[int, str] = _frozen(
    {
        3: "sim/flight_controls/brakes_1_auto",
        4: "sim/flight_controls/brakes_2_auto",
        0: "sim/flight_controls/brakes_rto_auto",
        5: "sim/flight_controls/brakes_rto_auto",
    }
)


def lookup(overrides: Plans, key: str, setting: str) -> tuple[str, Plan]:
    """("ok", plan), ("absent", ()) when the aircraft has no such switch, or
    ("unavailable", ()) when the switch has no such setting."""
    table = overrides[key] if key in overrides else GENERIC.get(key)
    if table is None:
        return "absent", ()
    plan = table.get(setting)
    if not plan:
        return "unavailable", ()
    return "ok", plan
