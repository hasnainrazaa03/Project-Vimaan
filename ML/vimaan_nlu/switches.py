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

import re
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
ICE = "sim/cockpit2/ice/"
EL = "sim/cockpit2/electrical/"
PUMP = "sim/cockpit2/engine/actuators/fuel_pump_on"  # per engine: 0 off, 1 on, 2 auto
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
ICES = ("pitot", "window", "wing", "engine", "prop")
# Part B keys per spoken thing: a generator or fuel pump by number, or all.
GENERATORS = {"1": ("power:generator1",), "2": ("power:generator2",)}
FUEL_PUMPS = {"1": ("fuel:left",), "2": ("fuel:right",)}
ALL_GENERATORS = ("power:generator1", "power:generator2")
ALL_FUEL_PUMPS = ("fuel:left", "fuel:center", "fuel:right")
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
        # --- part B: anti-ice, electrical, fuel pumps ---
        "ice:pitot": _on_off(
            ICE + "ice_pitot_heat_on_pilot", "sim/ice/pitot_heat0_on", "sim/ice/pitot_heat0_off"
        ),
        "ice:window": _on_off(
            ICE + "ice_window_heat_on", "sim/ice/window_heat_on", "sim/ice/window_heat_off"
        ),
        # X-Plane's own spelling, "surfce".
        "ice:wing": _on_off(
            ICE + "ice_surfce_heat_on", "sim/ice/wing_heat_on", "sim/ice/wing_heat_off"
        ),
        "ice:engine": _frozen(
            {
                setting: tuple(
                    Button(
                        f"{ICE}ice_inlet_heat_on_per_engine[{i}]",
                        f"sim/ice/inlet_heat{i}_{setting}",
                        want,
                    )
                    for i in (0, 1)
                )
                for setting, want in (("on", 1), ("off", 0))
            }
        ),
        # All props at once; ice_prop_heat_on is engine 1's (offered on propeller aircraft only).
        "ice:prop": _on_off(
            ICE + "ice_prop_heat_on", "sim/ice/prop_heat_on", "sim/ice/prop_heat_off"
        ),
        "power:battery": _on_off(
            EL + "battery_on[0]", "sim/electrical/battery_1_on", "sim/electrical/battery_1_off"
        ),
        "power:avionics": _on_off(
            SW + "avionics_power_on", "sim/systems/avionics_on", "sim/systems/avionics_off"
        ),
        "power:generator1": _on_off(
            EL + "generator_on[0]",
            "sim/electrical/generator_1_on",
            "sim/electrical/generator_1_off",
        ),
        "power:generator2": _on_off(
            EL + "generator_on[1]",
            "sim/electrical/generator_2_on",
            "sim/electrical/generator_2_off",
        ),
        "fuel:left": _on_off(PUMP + "[0]", "sim/fuel/fuel_pump_1_on", "sim/fuel/fuel_pump_1_off"),
        "fuel:right": _on_off(PUMP + "[1]", "sim/fuel/fuel_pump_2_on", "sim/fuel/fuel_pump_2_off"),
        "fuel:center": None,
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
        "ice:pitot": _frozen(
            {
                "on": (
                    Button(ICE + "ice_pitot_heat_on_pilot", "sim/ice/pitot_heat0_on", 1),
                    Button(ICE + "ice_pitot_heat_on_copilot", "sim/ice/pitot_heat1_on", 1),
                ),
                "off": (
                    Button(ICE + "ice_pitot_heat_on_pilot", "sim/ice/pitot_heat0_off", 0),
                    Button(ICE + "ice_pitot_heat_on_copilot", "sim/ice/pitot_heat1_off", 0),
                ),
            }
        ),
        # Boots AUTO / OFF: it replaces wing_boot*, "on" setting 2 (single,
        # auto-deflating) for wing and tail, "off" 0.
        "ice:wing": _frozen(
            {
                "on": (Button(ICE + "ice_surface_boot_on", "sim/ice/wing_boot_on", 2),),
                "off": (Button(ICE + "ice_surface_boot_on", "sim/ice/wing_boot_off", 0),),
            }
        ),
        "ice:engine": None,
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
C90_WINDSHIELD_L = Knob(
    "laminar/c90/iceRain/switch/windshield_antiice_pos_L",
    "laminar/c90/iceRain/switch/windshield_antiice_dn_L",
    "laminar/c90/iceRain/switch/windshield_antiice_up_L",
    -1,
    1,
)
C90_WINDSHIELD_R = Knob(
    "laminar/c90/iceRain/switch/windshield_antiice_pos_R",
    "laminar/c90/iceRain/switch/windshield_antiice_dn_R",
    "laminar/c90/iceRain/switch/windshield_antiice_up_R",
    -1,
    1,
)
C90_GEN1 = Knob(
    "laminar/c90/electrical/switch/gen1_switch_pos",
    "laminar/c90/electrical/switch/generator1_dn",
    "laminar/c90/electrical/switch/generator1_up",
    -1,
    1,
)
C90_GEN2 = Knob(
    "laminar/c90/electrical/switch/gen2_switch_pos",
    "laminar/c90/electrical/switch/generator2_dn",
    "laminar/c90/electrical/switch/generator2_up",
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
        "ice:pitot": B58["ice:pitot"],
        # Windshield L / R: NORMAL (1) / OFF (0) / HIGH (-1), stepped.
        "ice:window": _frozen(
            {
                "on": (C90_WINDSHIELD_L.at(1), C90_WINDSHIELD_R.at(1)),
                "off": (C90_WINDSHIELD_L.at(0), C90_WINDSHIELD_R.at(0)),
            }
        ),
        # Boots: one 7-second cycle per press (the boot dataref reads 3 while it
        # runs); there is no latched on, and nothing to turn off.
        "ice:wing": _frozen(
            {
                "on": (
                    Button(
                        ICE + "ice_surface_boot_on",
                        "laminar/c90/iceRain/switch/surface_deice_up",
                        3,
                    ),
                )
            }
        ),
        # Engine anti-ice is the ice vanes; a write holds and moves the switch.
        "ice:engine": _written(ICE + "anti_ice_engine_air[0]", ICE + "anti_ice_engine_air[1]"),
        # Its own switch: its timer undoes a write to X-Plane's prop heat.
        "ice:prop": _on_off(
            "laminar/c90/iceRain/switch/prop_deice_pos",
            "laminar/c90/iceRain/switch/prop_deice_up",
            "laminar/c90/iceRain/switch/prop_deice_dn",
        ),
        # Generators RESET (1) / ON (0) / OFF (-1); RESET is never the target.
        "power:generator1": _frozen({"on": (C90_GEN1.at(0),), "off": (C90_GEN1.at(-1),)}),
        "power:generator2": _frozen({"on": (C90_GEN2.at(0),), "off": (C90_GEN2.at(-1),)}),
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
# The battery switch's guard (0 closed, 1 open; it animates). While it is
# closed the 737 holds the battery on every frame, so "battery off" opens it.
B738_BATTERY_GUARD = Button(
    "laminar/B738/button_switch/guard_cover_pos[0]", "laminar/B738/guard_cover00", 1
)
TANK_PUMPS = "sim/cockpit2/fuel/fuel_tank_pump_on"  # per tank: [0] left, [1] center, [2] right
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
        "ice:pitot": _frozen(  # probe heat A and B
            {
                "on": (
                    Button(
                        "laminar/B738/toggle_switch/capt_probes_pos",
                        "laminar/B738/toggle_switch/capt_probes_pos_on",
                        1,
                    ),
                    Button(
                        "laminar/B738/toggle_switch/fo_probes_pos",
                        "laminar/B738/toggle_switch/fo_probes_pos_on",
                        1,
                    ),
                ),
                "off": (
                    Button(
                        "laminar/B738/toggle_switch/capt_probes_pos",
                        "laminar/B738/toggle_switch/capt_probes_pos_off",
                        0,
                    ),
                    Button(
                        "laminar/B738/toggle_switch/fo_probes_pos",
                        "laminar/B738/toggle_switch/fo_probes_pos_off",
                        0,
                    ),
                ),
            }
        ),
        "ice:window": _frozen(  # L-FWD, R-FWD, L-SIDE, R-SIDE
            {
                setting: tuple(
                    Button(
                        f"{ICE}ice_window_heat_on_window[{i}]",
                        f"sim/ice/{name}_heat_{setting}",
                        want,
                    )
                    for i, name in enumerate(("window", "window2", "window3", "window4"))
                )
                for setting, want in (("on", 1), ("off", 0))
            }
        ),
        "ice:prop": None,
        "power:battery": _frozen(
            {
                "on": (Button(EL + "battery_on[0]", "sim/electrical/battery_1_on", 1),),
                "off": (
                    B738_BATTERY_GUARD,
                    Button(EL + "battery_on[0]", "sim/electrical/battery_1_off", 0),
                ),
            }
        ),
        "power:avionics": None,  # none: its avionics follow the buses every frame
        # GEN 1 / 2 are momentary: "dn" is ON, "up" is OFF.
        "power:generator1": _on_off(
            EL + "generator_on[0]", "laminar/B738/switch/gen1_dn", "laminar/B738/switch/gen1_up"
        ),
        "power:generator2": _on_off(
            EL + "generator_on[1]", "laminar/B738/switch/gen2_dn", "laminar/B738/switch/gen2_up"
        ),
        # Each tank's forward and aft pumps are one switch here.
        "fuel:left": _written(TANK_PUMPS + "[0]"),
        "fuel:center": _written(TANK_PUMPS + "[1]"),
        "fuel:right": _written(TANK_PUMPS + "[2]"),
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


A333_PROBE_WINDOW = _frozen(
    {
        "on": (
            Button(
                "laminar/A333/monitors/probe_window_heat",
                "laminar/A333/buttons/probe_window_heat_toggle",
                1,
            ),
        ),
        "auto": (
            Button(
                "laminar/A333/monitors/probe_window_heat",
                "laminar/A333/buttons/probe_window_heat_toggle",
                0,
            ),
        ),
    }
)


def _a333_pumps(*names: str) -> Mapping[str, Plan]:
    return _frozen(
        {
            setting: tuple(
                Button(
                    f"laminar/A333/fuel/buttons/{n}_pump_pos",
                    f"laminar/A333/switches/{n}_pump_toggle",
                    want,
                )
                for n in names
            )
            for setting, want in (("on", 1), ("off", 0))
        }
    )


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
        # One PROBE / WINDOW HEAT button, AUTO (0) or ON (1): there is no off.
        "ice:pitot": A333_PROBE_WINDOW,
        "ice:window": A333_PROBE_WINDOW,
        "ice:wing": _toggled("laminar/A333/buttons/wing_anti_ice", "sim/ice/wing_heat_tog"),
        "ice:engine": _frozen(
            {
                setting: tuple(
                    Button(
                        f"laminar/A333/buttons/engine_anti_ice{n}",
                        f"sim/ice/inlet_heat{n - 1}_tog",
                        want,
                    )
                    for n in (1, 2)
                )
                for setting, want in (("on", 1), ("off", 0))
            }
        ),
        "ice:prop": None,
        # Read the buttons: X-Plane's battery_on can read 0 with the button on.
        "power:battery": _frozen(
            {
                setting: (
                    Button(
                        "laminar/A333/buttons/battery1_pos", "sim/electrical/battery_1_toggle", want
                    ),
                    Button(
                        "laminar/A333/buttons/battery2_pos", "sim/electrical/battery_2_toggle", want
                    ),
                )
                for setting, want in (("on", 1), ("off", 0))
            }
        ),
        "power:avionics": None,
        "power:generator1": _toggled(
            "laminar/A333/buttons/gen1_pos", "sim/electrical/generator_1_toggle"
        ),
        "power:generator2": _toggled(
            "laminar/A333/buttons/gen2_pos", "sim/electrical/generator_2_toggle"
        ),
        # Its own pump buttons only: X-Plane's fuel_pump_on is its engine feed
        # (set by the engine master switches), so sim/fuel/fuel_pump_N_* could
        # starve a running engine.
        "fuel:left": _a333_pumps("left1", "left2"),
        "fuel:right": _a333_pumps("right1", "right2"),
        "fuel:center": _a333_pumps("center_left", "center_right"),
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

# --- Cessna Citation X (phase L.2) -------------------------------------------
# Every switch is its own (xlua CitX_lights, CitX_ice, CitX_electrical,
# CitX_fuel, CitX_autopilot), and its scripts write X-Plane's datarefs from
# them every frame, so its commands click them. A toggle is read on the
# X-Plane dataref it drives; a three-position switch on its own position
# dataref (the "_term" one; the other animates toward it).
CITX = "laminar/CitX/"
CITX_ANTI_COLL = Knob(  # 0 OFF, 1 GND REC (beacons), 2 GND REC + ANTI COLL (strobes)
    CITX + "lights/gnd_rec_anti_coll_term",
    CITX + "lights/cmd_gnd_rec_anti_coll_dwn",
    CITX + "lights/cmd_gnd_rec_anti_coll_up",
    0,
    2,
)
# -1 SEAT BELT, 0 OFF, 1 PASS SAFETY (seat belt and no smoking); its "up"
# command turns it toward -1.
CITX_SIGNS = Knob(
    CITX + "lights/seat_belt_pass_safety_term",
    CITX + "lights/cmd_seat_belt_pass_safety_up",
    CITX + "lights/cmd_seat_belt_pass_safety_dwn",
    -1,
    1,
)


def _citx_three(name: str, down: str, up: str, low: int, high: int) -> Knob:
    return Knob(f"{CITX}{name}_term", f"{CITX}{down}", f"{CITX}{up}", low, high)


# Windshield heat 0 OFF, 1 ON, 2 O'RIDE (sprung back to ON).
CITX_WINDSHIELD_L = _citx_three(
    "ice/windshield_left", "ice/cmd_windshield_left_dwn", "ice/cmd_windshield_left_up", 0, 2
)
CITX_WINDSHIELD_R = _citx_three(
    "ice/windshield_right", "ice/cmd_windshield_right_dwn", "ice/cmd_windshield_right_up", 0, 2
)
# Generators -1 RESET (sprung back to OFF), 0 OFF, 1 ON.
CITX_GEN_L = _citx_three(
    "electrical/generator_left",
    "electrical/cmd_generator_left_dwn",
    "electrical/cmd_generator_left_up",
    -1,
    1,
)
CITX_GEN_R = _citx_three(
    "electrical/generator_right",
    "electrical/cmd_generator_right_dwn",
    "electrical/cmd_generator_right_up",
    -1,
    1,
)
# Fuel boost pumps -1 NORM, 0 OFF, 1 ON.
CITX_BOOST_L = _citx_three(
    "fuel/boost_left", "fuel/cmd_boost_left_dwn", "fuel/cmd_boost_left_up", -1, 1
)
CITX_BOOST_R = _citx_three(
    "fuel/boost_right", "fuel/cmd_boost_right_dwn", "fuel/cmd_boost_right_up", -1, 1
)


def _toggled_pair(left: tuple[str, str], right: tuple[str, str]) -> Mapping[str, Plan]:
    """Two toggles (state, command) set together, left then right."""
    return _frozen(
        {
            setting: (Button(left[0], left[1], want), Button(right[0], right[1], want))
            for setting, want in (("on", 1), ("off", 0))
        }
    )


def _three_on_off(*knobs: Knob, on: int = 1, off: int = 0) -> Mapping[str, Plan]:
    return _frozen({"on": tuple(k.at(on) for k in knobs), "off": tuple(k.at(off) for k in knobs)})


CITX_PLANS: Plans = _frozen(
    {
        "light:landing": _toggled_pair(
            (SW + "landing_lights_switch[0]", CITX + "lights/cmd_landing_left_toggle"),
            (SW + "landing_lights_switch[1]", CITX + "lights/cmd_landing_right_toggle"),
        ),
        "light:taxi": _toggled(SW + "taxi_light_on", CITX + "lights/cmd_taxi_toggle"),
        "light:nav": _toggled(SW + "navigation_lights_on", CITX + "lights/cmd_navigation_toggle"),
        "light:beacon": _frozen({"on": (CITX_ANTI_COLL.at(1, 2),), "off": (CITX_ANTI_COLL.at(0),)}),
        # Strobes off leaves the beacons (GND REC) on.
        "light:strobe": _frozen({"on": (CITX_ANTI_COLL.at(2),), "off": (CITX_ANTI_COLL.at(0, 1),)}),
        "light:logo": _toggled(
            SW + "generic_lights_switch[0]", CITX + "lights/cmd_tail_flood_toggle"
        ),
        "light:wing": _toggled(SW + "generic_lights_switch[1]", CITX + "ice/cmd_wing_light_toggle"),
        "sign:seat belt": _frozen(
            {
                "on": (Check(SW + "fasten_seat_belts", 1), CITX_SIGNS.at(-1)),
                "off": (CITX_SIGNS.at(0),),
            }
        ),
        # No smoking alone isn't a position: off leaves the seat belt sign.
        "sign:no smoking": _frozen(
            {
                "on": (CITX_SIGNS.at(1),),
                "off": (Check(SW + "no_smoking", 0), CITX_SIGNS.at(-1)),
            }
        ),
        "yaw_damper": _toggled(SW + "yaw_damper_on", CITX + "autopilot/cmd_yd_toggle"),
        # Pitot and static L / R (each also heats its static port and AoA vane).
        "ice:pitot": _toggled_pair(
            (ICE + "ice_pitot_heat_on_pilot", CITX + "ice/cmd_pitot_static_left_toggle"),
            (ICE + "ice_pitot_heat_on_copilot", CITX + "ice/cmd_pitot_static_right_toggle"),
        ),
        "ice:window": _three_on_off(CITX_WINDSHIELD_L, CITX_WINDSHIELD_R),
        # Wing anti-ice is its slat bleed air.
        "ice:wing": _toggled(ICE + "ice_surface_hot_bleed_air_on", CITX + "ice/cmd_slat_toggle"),
        "ice:engine": _toggled_pair(
            (ICE + "cowling_thermal_anti_ice_per_engine[0]", CITX + "ice/cmd_engine_left_toggle"),
            (
                ICE + "cowling_thermal_anti_ice_per_engine[1]",
                CITX + "ice/cmd_engine_right_toggle",
            ),
        ),
        "ice:prop": None,
        "power:battery": _toggled_pair(
            (CITX + "electrical/battery_left_term", CITX + "electrical/cmd_battery_left_toggle"),
            (CITX + "electrical/battery_right_term", CITX + "electrical/cmd_battery_right_toggle"),
        ),
        "power:avionics": _toggled(
            CITX + "electrical/avionics_term", CITX + "electrical/cmd_avionics_toggle"
        ),
        "power:generator1": _three_on_off(CITX_GEN_L),
        "power:generator2": _three_on_off(CITX_GEN_R),
        "fuel:left": _three_on_off(CITX_BOOST_L),
        "fuel:right": _three_on_off(CITX_BOOST_R),
    }
)

# --- McDonnell Douglas MD-82 (phase L.2) -------------------------------------
# Its own commands move X-Plane's datarefs one click at a time (xlua
# md80_swithces, md80_ice); the rest of its cockpit clicks X-Plane's commands.
MD82_CMD = "laminar/md82cmd/switches/"


def _md82(state: str, name: str, low: int, high: int) -> Knob:
    """Its "up" command turns the switch toward the lower value."""
    return Knob(state, f"{MD82_CMD}{name}_up", f"{MD82_CMD}{name}_dwn", low, high)


# Landing lights L / R: -1 RETRACT, 0 EXTEND (off), 1 ON.
MD82_LANDING_L = _md82(SW + "landing_lights_switch[1]", "landing_lights_switch_L", -1, 1)
MD82_LANDING_R = _md82(SW + "landing_lights_switch[2]", "landing_lights_switch_R", -1, 1)
MD82_NOSE = _md82(SW + "taxi_light_on", "landing_lights_switch_nose", 0, 2)  # 0 OFF, 1, 2
MD82_NAVSTROBE = _md82(  # 0 OFF, 1 POS, 2 POS + STROBE
    "laminar/md82/switches/navstrobe_lights_switch", "navstrobe_lights_switch", 0, 2
)
MD82_SEATBELT = _md82(SW + "fasten_seat_belts", "seatbelts_switch", 0, 2)  # 0 OFF, 1 AUTO, 2 ON
MD82_NO_SMOKING = _md82(SW + "no_smoking", "nosmoke_switch", 0, 2)  # 0 OFF, 1 AUTO, 2 ON
# The pitot / stall heat selector: 0 OFF, 1-9 heat every probe (the position
# picks the probe its meter shows); it wraps from 9 to 0.
MD82_PITOT = Knob(
    "laminar/md82/ice/heatknob",
    "laminar/md82cmd/ice/selheatknob_dwn",
    "laminar/md82cmd/ice/selheatknob_up",
    0,
    9,
)
MD82_PLANS: Plans = _frozen(
    {
        "light:landing": _frozen(
            {
                "on": (MD82_LANDING_L.at(1), MD82_LANDING_R.at(1)),
                "off": (MD82_LANDING_L.at(-1, 0), MD82_LANDING_R.at(-1, 0)),
            }
        ),
        "light:taxi": _frozen({"on": (MD82_NOSE.at(1, 2),), "off": (MD82_NOSE.at(0),)}),
        "light:nav": _frozen({"on": (MD82_NAVSTROBE.at(1, 2),), "off": (MD82_NAVSTROBE.at(0),)}),
        # Strobes off leaves the position lights on.
        "light:strobe": _frozen({"on": (MD82_NAVSTROBE.at(2),), "off": (MD82_NAVSTROBE.at(0, 1),)}),
        "light:logo": _toggled(SW + "generic_lights_switch[0]", "sim/lights/generic_01_light_tog"),
        "light:wing": _toggled(SW + "generic_lights_switch[3]", "sim/lights/generic_04_light_tog"),
        "sign:seat belt": _three(MD82_SEATBELT),
        "sign:no smoking": _three(MD82_NO_SMOKING),
        "ice:pitot": _frozen({"on": (MD82_PITOT.at(1, 9),), "off": (MD82_PITOT.at(0),)}),
        # Left and right wing heat are separate switches.
        "ice:wing": _frozen(
            {
                setting: (
                    Button(f"{ICE}ice_surfce_heat_left_on", f"sim/ice/wing_heat0_{setting}", want),
                    Button(f"{ICE}ice_surfce_heat_right_on", f"sim/ice/wing_heat1_{setting}", want),
                )
                for setting, want in (("on", 1), ("off", 0))
            }
        ),
        # No avionics master: its electrical script sets avionics power from
        # the buses (md80_electrical), as on the 737.
        "power:avionics": None,
        # Tank pumps: [0] center, [1] left, [2] right (its cockpit's labels).
        "fuel:left": _on_off(
            TANK_PUMPS + "[1]", "sim/fuel/fuel_tank_pump_2_on", "sim/fuel/fuel_tank_pump_2_off"
        ),
        "fuel:center": _on_off(
            TANK_PUMPS + "[0]", "sim/fuel/fuel_tank_pump_1_on", "sim/fuel/fuel_tank_pump_1_off"
        ),
        "fuel:right": _on_off(
            TANK_PUMPS + "[2]", "sim/fuel/fuel_tank_pump_3_on", "sim/fuel/fuel_tank_pump_3_off"
        ),
    }
)

# --- Cirrus SR22 and Vision SF50 (phase L.2) --------------------------------
# The SR22's boost pump: -1 PRIME, 0 OFF, 1 BOOST (xlua SR22.systems).
SR22_BOOST = Knob(
    "laminar/sr22/switch/boost_pump",
    "laminar/sr22/switch/boost_pump_dn",
    "laminar/sr22/switch/boost_pump_up",
    -1,
    1,
)
SR22_PLANS: Plans = _frozen(
    {
        "fuel:left": _frozen({"on": (SR22_BOOST.at(1),), "off": (SR22_BOOST.at(0),)}),
        # Its TKS fluid system (wings, tail and prop) is the wing anti-ice.
        "ice:wing": _on_off(
            "laminar/sr22/switch/ice_protect",
            "laminar/sr22/switch/ice_protect_on",
            "laminar/sr22/switch/ice_protect_off",
        ),
    }
)
# The SF50 has no fuel pump switch: its pump follows the ignition knob
# (xlua sf50_ignition), so "fuel pumps off" would be the engine's fuel.
SF50_PLANS: Plans = _frozen({"fuel:left": None, "fuel:right": None})

# --- What an aircraft's cockpit shows (phase L.2) ---------------------------
# A switch is clicked in an aircraft's cockpit objects or driven by its
# scripts, so their .obj and .lua files name its command or dataref. Where no
# such file names any of these, the aircraft has no such switch: Vimaan says
# so instead of setting X-Plane's value for a switch that isn't there (the
# C172's seat belt sign, the SR22's taxi light). Checked against the Tier 3
# findings (X-Plane 12.4.4, 2026-11-21): on the four proven aircraft the
# switches this finds missing are the ones their overrides already leave out.
# The .acf isn't read: it names every system (its electrical loads).
SHOWN_BY: Mapping[str, tuple[bytes, ...]] = _frozen(
    {
        "light:landing": (b"landing_light", b"landing_0"),
        "light:taxi": (b"taxi_light",),
        "light:beacon": (b"beacon_light", b"beacon_on"),
        "light:strobe": (b"strobe_light",),
        "light:nav": (b"nav_light", b"navigation_light"),
        "sign:seat belt": (b"seatbelt", b"seat_belt"),
        "sign:no smoking": (b"no_smoking",),
        "yaw_damper": (b"yaw_damp",),
        "ice:pitot": (b"pitot_heat",),
        "ice:window": (b"window_heat",),
        "ice:wing": (b"wing_heat", b"surfce_heat", b"wing_boot", b"surface_boot"),
        "ice:engine": (b"inlet_heat", b"anti_ice_engine"),
        "ice:prop": (b"prop_heat",),
        "power:avionics": (b"avionics_on", b"avionics_off", b"avionics_power", b"avionics_tog"),
        "fuel:left": (b"fuel_pump",),
        "fuel:right": (b"fuel_pump",),
        "autobrake": (b"auto_brake", b"autobrake", b"brakes_up_auto"),
    }
)


def settings_of(overrides: Plans, key: str) -> tuple[str, ...]:
    """The settings switch ``key`` has here, in a stable order."""
    table = overrides[key] if key in overrides else GENERIC.get(key)
    order = dict.fromkeys(("on", "off", "auto", *SPEEDBRAKE_SETTINGS, *AUTOBRAKE_SETTINGS))
    return tuple(s for s in order if table and s in table)


# Per-engine and per-generator datarefs, "name[i]": a step for engine (or
# generator) i is dropped on an aircraft with fewer. Generators are counted on
# their own: the SR22 and SF50 have two on their one engine.
_ENGINE_INDEXED = re.compile(r"(per_engine|fuel_pump_on|generator_on)\[(\d+)\]$")


def for_engines(plan: Plan, engines: int | None, generators: int | None = None) -> Plan:
    out = []
    for step in plan:
        m = next(
            (_ENGINE_INDEXED.search(n) for n in names(step) if _ENGINE_INDEXED.search(n)), None
        )
        count = (generators or engines) if m and m.group(1) == "generator_on" else engines
        if m is None or not count or int(m.group(2)) < count:
            out.append(step)
    return tuple(out)


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
