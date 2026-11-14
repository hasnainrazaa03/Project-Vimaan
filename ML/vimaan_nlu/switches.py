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


def settings_of(overrides: Plans, key: str) -> tuple[str, ...]:
    """The settings switch ``key`` has here, in a stable order."""
    table = overrides[key] if key in overrides else GENERIC.get(key)
    order = dict.fromkeys(("on", "off", "auto", *SPEEDBRAKE_SETTINGS, *AUTOBRAKE_SETTINGS))
    return tuple(s for s in order if table and s in table)


# Per-engine (or per-generator) datarefs, "name[i]": a step for engine i is
# dropped on an aircraft with fewer engines.
_ENGINE_INDEXED = re.compile(r"(per_engine|fuel_pump_on|generator_on)\[(\d+)\]$")


def for_engines(plan: Plan, engines: int | None) -> Plan:
    if not engines:
        return plan
    out = []
    for step in plan:
        m = next(
            (_ENGINE_INDEXED.search(n) for n in names(step) if _ENGINE_INDEXED.search(n)), None
        )
        if m is None or int(m.group(2)) < engines:
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
