"""Which aircraft is loaded, what it can do, and which controls drive it.

Nothing here is keyed to an aircraft's name. A profile is resolved in layers,
most general first (docs/PHASE_PLAN.md, Tier 1, "How it generalizes"):

1. **X-Plane's standard controls** (``DEFAULT_COMMANDS``), which default and
   most add-on aircraft honour.
2. **Capabilities X-Plane reports** when the aircraft loads (``AircraftInfo``):
   engine count and type, flap detents and angles, retractable gear,
   autopilot type, V-speeds.
3. **Overrides that apply only when the loaded aircraft has the custom
   control they use** (``OVERRIDES``). The A330's A/THR button is used only if
   ``laminar/A333/autopilot/a_thr_toggle`` exists. ICAO codes and author
   strings are not used for matching: Laminar's 737 and A330 list individual
   authors, and add-ons reuse ICAO codes.
4. **Probe results** measured on this aircraft (``vimaan_nlu.probe``).
5. **Verification after every action**, in the plugin: on an aircraft no one
   has probed, a control it ignores is reported, never assumed to work.

Pure Python (no XPPython3): the plugin reads the datarefs and passes them in.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

# sim/aircraft/prop/acf_en_type (X-Plane 12.4.3 DataRefs.txt). Values the
# list doesn't name are "unknown", which starts like a piston (hold the
# starter until it runs) but is logged as unknown.
ENGINE_KINDS: Mapping[int, str] = MappingProxyType(
    {
        0: "piston",  # recip carb
        1: "piston",  # recip injected
        3: "electric",
        5: "jet",  # single spool
        6: "rocket",
        7: "jet",  # multi spool
        9: "turboprop",  # free turbine
        10: "turboprop",  # fixed turbine
    }
)

# sim/aircraft/autopilot/preconfigured_ap_type
AUTOPILOT_TYPES: Mapping[int, str] = MappingProxyType(
    {
        0: "X-Plane custom",
        1: "Airliner",
        2: "GFC-700",
        3: "S-Tec 55",
        4: "S-Tec 55 with altitude preselect",
        5: "KAP-140 single axis",
        6: "KAP-140 dual axis",
        7: "KAP-140 dual axis with altitude preselect",
        8: "Piper Autocontrol",
        9: "Rockwell/Collins FCS-65",
    }
)

# Layer 1: the standard control for each role. Handlers ask the profile for
# a role, never for a name, so an override can replace it.
DEFAULT_COMMANDS: Mapping[str, str | None] = MappingProxyType(
    {
        "autopilot_toggle": "sim/autopilot/servos_toggle",
        "autopilot_on": "sim/autopilot/servos_on",
        "autopilot2_toggle": "sim/autopilot/servos2_toggle",
        "autopilot2_on": "sim/autopilot/servos2_on",
        "mode_heading": "sim/autopilot/heading",
        "mode_nav": "sim/autopilot/NAV",
        "mode_approach": "sim/autopilot/approach",
        "mode_back_course": "sim/autopilot/back_course",
        "mode_altitude_hold": "sim/autopilot/altitude_hold",
        "mode_altitude_arm": "sim/autopilot/altitude_arm",
        "mode_vertical_speed": "sim/autopilot/vertical_speed",
        "mode_level_change": "sim/autopilot/level_change",
        "mode_vnav": "sim/autopilot/vnav",
        "mode_gpss": "sim/autopilot/gpss",
        "knots_mach_toggle": "sim/autopilot/knots_mach_toggle",
        "autothrottle_on": "sim/autopilot/autothrottle_on",
        "autothrottle_off": "sim/autopilot/autothrottle_off",
        "autothrottle_toggle": "sim/autopilot/autothrottle_toggle",
        "baro_std": "sim/instruments/barometer_std",
        # Opening an engine's fuel during a turbine start. None: write the
        # sim's fuel lever (mixture ratio) instead of pressing a command.
        "fuel_on_1": None,
        "fuel_on_2": None,
        # Pressed after writing a target on aircraft with selected/managed
        # targets (Airbus FCU). None: the standard sim has no such step.
        "speed_select": None,
        "heading_select": None,
        "altitude_select": None,
        "vertical_select": None,
    }
)


@dataclass(frozen=True)
class Override:
    """Layer 3: replaces roles, only when every ``requires`` command exists."""

    name: str
    requires: tuple[str, ...]
    commands: Mapping[str, str | None]
    source: str  # where the finding comes from


OVERRIDES: tuple[Override, ...] = (
    Override(
        name="airbus_fcu",
        requires=(
            "laminar/A333/autopilot/speed_knob_pull",
            "laminar/A333/autopilot/a_thr_toggle",
        ),
        commands=MappingProxyType(
            {
                # A/THR is its own button; the sim's autothrottle commands
                # don't drive the A330's FADEC logic.
                "autothrottle_toggle": "laminar/A333/autopilot/a_thr_toggle",
                "autothrottle_on": None,
                "autothrottle_off": None,
                # Pull = selected (fly the dialled value), push = managed.
                "speed_select": "laminar/A333/autopilot/speed_knob_pull",
                "heading_select": "laminar/A333/autopilot/heading_knob_pull",
                "altitude_select": "laminar/A333/autopilot/altitude_knob_pull",
                "vertical_select": "laminar/A333/autopilot/vertical_knob_pull",
            }
        ),
        source="Laminar A330-300 xlua A333.autopilot: create_command(...), X-Plane 12.4.3",
    ),
    Override(
        name="b738_condition_levers",
        requires=(
            "laminar/B738/engines/condition_lever_1_idle",
            "laminar/B738/engines/condition_lever_2_idle",
        ),
        commands=MappingProxyType(
            {
                # The 737's own levers drive its fuel cutoff; a direct write to
                # the sim lever may be overwritten by its xlua each frame.
                "fuel_on_1": "laminar/B738/engines/condition_lever_1_idle",
                "fuel_on_2": "laminar/B738/engines/condition_lever_2_idle",
            }
        ),
        source="Laminar 737-800 xlua B738 scripts: create_command(...), X-Plane 12.4.3",
    ),
    Override(
        name="b738_efis_baro",
        requires=("laminar/B738/EFIS_control/capt/push_button/std_press",),
        commands=MappingProxyType(
            {"baro_std": "laminar/B738/EFIS_control/capt/push_button/std_press"}
        ),
        source="Laminar 737-800 xlua B738.glareshield: create_command(...), X-Plane 12.4.3",
    ),
)


@dataclass(frozen=True)
class AircraftInfo:
    """Layer 2: what X-Plane says about the loaded aircraft. Any field may be
    None when its dataref is missing or unreadable."""

    icao: str = ""
    author: str = ""
    description: str = ""
    acf_path: str = ""
    num_engines: int | None = None
    engine_types: tuple[int, ...] = ()
    flap_detents: int | None = None
    flap_degrees: tuple[float, ...] = ()
    gear_retractable: bool | None = None
    autopilot_type: int | None = None
    single_axis_autopilot: bool | None = None
    vso_kt: float | None = None
    vfe_kt: float | None = None
    vno_kt: float | None = None
    vne_kt: float | None = None
    mmo: float | None = None

    @property
    def engine_kind(self) -> str:
        """piston / turboprop / jet / electric / rocket / unknown, from the
        first engine (mixed types are logged as "mixed")."""
        kinds = {ENGINE_KINDS.get(t, "unknown") for t in self.engine_types}
        if not kinds:
            return "unknown"
        if len(kinds) > 1:
            return "mixed"
        return kinds.pop()

    @property
    def autopilot_name(self) -> str:
        if self.autopilot_type is None:
            return "unknown"
        return AUTOPILOT_TYPES.get(self.autopilot_type, f"type {self.autopilot_type}")

    @property
    def key(self) -> str:
        """Stable file-name key: the aircraft folder and .acf file
        ("Airbus A330-300/A330.acf" -> "Airbus_A330-300__A330"), so results
        survive moving X-Plane; falls back to the ICAO code."""
        parts = [p for p in re.split(r"[\\/]", self.acf_path) if p][-2:]
        if parts:
            parts[-1] = re.sub(r"\.acf$", "", parts[-1], flags=re.IGNORECASE)
            raw = "__".join(parts)
        else:
            raw = self.icao or "unknown"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)

    def flap_notch_degrees(self) -> tuple[float, ...]:
        """Modelled surface angle at each handle position, up first:
        ``acf_flap_dn[0]`` is 0 (up) and ``[1..detents]`` the notches (737:
        0, 3, 5, 8, 10, 12, 14, 18, 40). These are *not* the handle labels
        (the 737's "flaps 15" is notch 5 at 12 degrees), so they can only
        name a notch on aircraft whose labels are degrees (Baron: 13, 30)."""
        if not self.flap_detents:
            return ()
        angles = tuple(float(a) for a in self.flap_degrees[: self.flap_detents + 1])
        return angles if len(angles) == self.flap_detents + 1 and angles[0] == 0 else ()


@dataclass(frozen=True)
class Profile:
    """How Vimaan drives this aircraft."""

    info: AircraftInfo
    commands: Mapping[str, str | None]
    overrides: tuple[str, ...] = ()
    # step -> "works" | "no_change" | "missing", from the probe (layer 4)
    probed: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def engine_start(self) -> str:
        kind = self.info.engine_kind
        return kind if kind in ("piston", "jet", "turboprop", "electric") else "piston"

    def command(self, role: str) -> str | None:
        return self.commands.get(role)

    def supports(self, role: str) -> bool | None:
        """True / False when measured, None when nobody has probed it."""
        verdict = self.probed.get(role)
        if verdict is None:
            return None if self.command(role) else False
        return verdict == "works"

    def summary(self) -> dict[str, Any]:
        """For the ``aircraft`` telemetry event."""
        info = self.info
        return {
            "icao": info.icao,
            "author": info.author,
            "acf": info.acf_path,
            "key": info.key,
            "engines": info.num_engines,
            "engine_kind": info.engine_kind,
            "engine_start": self.engine_start,
            "flap_detents": info.flap_detents,
            "flap_notch_degrees": list(info.flap_notch_degrees()),
            "gear_retractable": info.gear_retractable,
            "autopilot": info.autopilot_name,
            "single_axis_autopilot": info.single_axis_autopilot,
            "v_speeds": {
                "vso": info.vso_kt,
                "vfe": info.vfe_kt,
                "vno": info.vno_kt,
                "vne": info.vne_kt,
                "mmo": info.mmo,
            },
            "overrides": list(self.overrides),
            "probed_steps": len(self.probed),
        }


def resolve(
    info: AircraftInfo,
    has_command: Callable[[str], bool],
    probed: Mapping[str, str] | None = None,
) -> Profile:
    """Layers 1-4 -> the profile the plugin uses for this aircraft."""
    commands: dict[str, str | None] = dict(DEFAULT_COMMANDS)
    applied = []
    for override in OVERRIDES:
        if all(has_command(name) for name in override.requires):
            commands.update(override.commands)
            applied.append(override.name)
    return Profile(
        info=info,
        commands=MappingProxyType(commands),
        overrides=tuple(applied),
        probed=MappingProxyType(dict(probed or {})),
    )


def override_names() -> tuple[str, ...]:
    """Every aircraft-specific control the overrides use (reference checker)."""
    names = {n for o in OVERRIDES for n in o.requires}
    names |= {v for o in OVERRIDES for v in o.commands.values() if v}
    return tuple(sorted(names))
