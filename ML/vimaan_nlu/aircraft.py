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

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from vimaan_nlu import switches
from vimaan_nlu.switches import Knob

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
        # Press-to-engage buttons on aircraft that track the autopilot in
        # their own logic (737 CMD A / B). None: use the checked toggle.
        "autopilot_engage": None,
        "autopilot2_engage": None,
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
        # Arms an autothrottle that is hard off (autothrottle_enabled -1) so
        # it can engage: the 737's A/T ARM switch is this command (its cockpit
        # object), as are other X-Plane autothrottles with an arm switch.
        "autothrottle_arm": "sim/autopilot/autothrottle_arm",
        "baro_std": "sim/instruments/barometer_std",
        # One step of the altimeter knob, for aircraft whose own logic owns
        # the setting (the 737 put written values back; its scripts handle
        # these commands).
        "baro_up": "sim/instruments/barometer_up",
        "baro_down": "sim/instruments/barometer_down",
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
        # Transponder ident. None: X-Plane's sim/transponder/transponder_ident.
        "xpdr_ident": None,
    }
)


@dataclass(frozen=True)
class Override:
    """Layer 3: replaces roles, only when every ``requires`` command exists
    and every ``live`` dataref is registered.

    ``live`` names a dataref the aircraft's own scripts create. Their commands
    stay registered for the rest of the X-Plane session after the aircraft is
    unloaded, but their datarefs go with it: on 2026-11-12 an A330 loaded
    after the 737 picked up the 737's overrides and pressed its MCP buttons."""

    name: str
    requires: tuple[str, ...]
    commands: Mapping[str, str | None]
    source: str  # where the finding comes from
    # Non-command facts, e.g. flap handle labels.
    settings: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    live: tuple[str, ...] = ()


# Datarefs each aircraft's own scripts create (create_dataref in its xlua),
# present only while it's loaded.
B738_LIVE = "laminar/B738/autopilot/cmd_a_status"
A333_LIVE = "laminar/A333/autopilot/ap1_pos"

# Transponder selectors (positions set by Override.settings["xpdr_knobs"]).
B738_XPDR = Knob(
    "laminar/B738/knob/transponder_pos",
    "laminar/B738/knob/transponder_mode_dn",
    "laminar/B738/knob/transponder_mode_up",
    0,
    5,
)
A333_AUTO_ON = Knob(  # -1 STBY, 0 AUTO, 1 ON
    "laminar/A333/transponder/auto_on_knob_pos",
    "laminar/A333/transponder/auto_on_off_left",
    "laminar/A333/transponder/auto_on_off_right",
    -1,
    1,
)
A333_ALT_RPT = Knob(  # 0 off, 1 on
    "laminar/A333/transponder/alt_rpt_knob_pos",
    "laminar/A333/transponder/alt_rpt_off",
    "laminar/A333/transponder/alt_rpt_on",
    0,
    1,
)
A333_TA_RA = Knob(  # TCAS: 0 STBY, 1 TA, 2 TA/RA
    "laminar/A333/transponder/ta_ra_knob_pos",
    "laminar/A333/transponder/ta_ra_left",
    "laminar/A333/transponder/ta_ra_right",
    0,
    2,
)

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
                "autothrottle_arm": None,
                # Pull = selected (fly the dialled value), push = managed.
                "speed_select": "laminar/A333/autopilot/speed_knob_pull",
                "heading_select": "laminar/A333/autopilot/heading_knob_pull",
                "altitude_select": "laminar/A333/autopilot/altitude_knob_pull",
                "vertical_select": "laminar/A333/autopilot/vertical_knob_pull",
                # Managed modes: heading knob push engages GPS steering (NAV),
                # altitude knob push the FMS's vertical mode (sim/autopilot/FMS).
                "mode_gpss": "laminar/A333/autopilot/heading_knob_push",
                "mode_vnav": "laminar/A333/autopilot/altitude_knob_push",
            }
        ),
        source="Laminar A330-300 xlua A333.autopilot: create_command(...) and its "
        "*_CMDhandler functions, X-Plane 12.4.3",
        live=(A333_LIVE,),
        settings=MappingProxyType(
            {
                # CONF 1, 2, 3, FULL (4 detents); "full" is its own word.
                "flap_labels": ("1", "2", "3"),
                # Managed vertical is FMS VNAV (fms_vnav), not the sim's
                # geometric-descent vnav_status.
                "status_datarefs": MappingProxyType(
                    {"mode_vnav": "sim/cockpit2/autopilot/fms_vnav"}
                ),
                # A/THR flips autothrottle_enabled between -1 (off) and 0
                # (armed); armed counts as on. While both thrust levers are at
                # FLX/MCT or TOGA its script holds it at armed every frame.
                "athr_on_from": 0,
                "athr_hint": "A/THR stays armed while the thrust levers are at FLX, MCT "
                "or TOGA. Move them to CL",
            }
        ),
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
        live=(B738_LIVE,),
        # Handle labels, notch 1..8 (POH); X-Plane's modelled angles differ
        # (notch 5 is 12 degrees), so they can't be derived. Used only when
        # the aircraft reports 8 detents.
        settings=MappingProxyType({"flap_labels": ("1", "2", "5", "10", "15", "25", "30", "40")}),
    ),
    Override(
        name="b738_mcp",
        requires=(
            "laminar/B738/autopilot/cmd_a_press",
            "laminar/B738/autopilot/hdg_sel_press",
        ),
        commands=MappingProxyType(
            {
                # The 737's MCP buttons call the same sim commands but also
                # update its own state (CMD_status / AP_status), which drives
                # its CMD light and annunciators; the generic commands skip
                # that, so its CMD A status read off while the servos flew.
                "autopilot_engage": "laminar/B738/autopilot/cmd_a_press",
                "autopilot2_engage": "laminar/B738/autopilot/cmd_b_press",
                "mode_heading": "laminar/B738/autopilot/hdg_sel_press",
                "mode_nav": "laminar/B738/autopilot/vorloc_press",
                "mode_approach": "laminar/B738/autopilot/app_press",
                "mode_altitude_hold": "laminar/B738/autopilot/alt_hld_press",
                "mode_vertical_speed": "laminar/B738/autopilot/vs_press",
                "mode_level_change": "laminar/B738/autopilot/lvl_chg_press",
                "mode_vnav": "laminar/B738/autopilot/vnav_press",
                "mode_gpss": "laminar/B738/autopilot/lnav_press",
            }
        ),
        source="Laminar 737-800 xlua B738.glareshield: create_command(...) and its "
        "*_CMDhandler functions, X-Plane 12.4.3",
        live=(B738_LIVE,),
    ),
    Override(
        name="b738_efis_baro",
        requires=("laminar/B738/EFIS_control/capt/push_button/std_press",),
        commands=MappingProxyType(
            {"baro_std": "laminar/B738/EFIS_control/capt/push_button/std_press"}
        ),
        source="Laminar 737-800 xlua B738.glareshield: create_command(...), X-Plane 12.4.3",
        live=(B738_LIVE,),
    ),
    Override(
        # The selector is set only by its own knob commands: a written mode
        # holds but leaves the knob where it was. 0 TEST, 1 STBY, 2 ALT RPTG
        # OFF, 3 XPNDR, 4 TA ONLY, 5 TA/RA. X-Plane's mode reads 2 for all of
        # 2-5, so the knob is what Vimaan checks. There is no off.
        name="b738_transponder",
        requires=(
            "laminar/B738/knob/transponder_mode_up",
            "laminar/B738/knob/transponder_mode_dn",
            "laminar/B738/push_button/transponder_ident_dn",
        ),
        commands=MappingProxyType({"xpdr_ident": "laminar/B738/push_button/transponder_ident_dn"}),
        settings=MappingProxyType(
            {
                "xpdr_knobs": {
                    "standby": (B738_XPDR.at(1),),
                    "on": (B738_XPDR.at(2, 5),),  # replying, any of these
                    "alt": (B738_XPDR.at(3, 5),),  # replying with altitude
                }
            }
        ),
        source="Laminar 737-800 xlua B738.systems: create_command(...) and its "
        "transponder knob handlers, X-Plane 12.4.3",
        live=(B738_XPDR.position,),
    ),
    Override(
        # A333.transponder recomputes X-Plane's mode every frame from three
        # selectors (and power, gear, airspeed), so a written mode is
        # overwritten: turn the selectors. AUTO (0) is the normal position
        # and replies once airborne or rolling; STBY (-1) is standby; there
        # is no off. TCAS STBY (0) also leaves the transponder in standby.
        name="a333_transponder",
        requires=(
            A333_AUTO_ON.down,
            A333_AUTO_ON.up,
            A333_ALT_RPT.down,
            A333_ALT_RPT.up,
            A333_TA_RA.down,
            A333_TA_RA.up,
        ),
        commands=MappingProxyType({}),
        settings=MappingProxyType(
            {
                "xpdr_knobs": {
                    "standby": (A333_AUTO_ON.at(-1),),
                    "on": (A333_AUTO_ON.at(0, 1), A333_TA_RA.at(1, 2)),
                    "alt": (A333_AUTO_ON.at(0, 1), A333_ALT_RPT.at(1), A333_TA_RA.at(1, 2)),
                }
            }
        ),
        source="Laminar A330-300 xlua A333.transponder: create_command(...) and "
        "A333_transponder_mode_switching, X-Plane 12.4.3",
        live=(A333_AUTO_ON.position,),
    ),
    # Tier 3 switches (vimaan_nlu.switches; findings in AIRCRAFT_PROFILES.md).
    Override(
        name="b58_switches",
        requires=("laminar/b58/surface_ice_switch_up", "laminar/b58/surface_ice_switch_dn"),
        commands=MappingProxyType({}),
        settings=MappingProxyType({"switches": switches.B58}),
        source="Laminar Baron 58 cockpit .obj manipulators and xlua B58_01_logic, X-Plane 12.4.3",
        live=("laminar/b58/inst_press_ind_pos",),
    ),
    Override(
        name="c90_switches",
        requires=(switches.C90_SIGNS.down, switches.C90_SIGNS.up),
        commands=MappingProxyType({}),
        settings=MappingProxyType({"switches": switches.C90}),
        source="Laminar King Air C90B cockpit .obj manipulators and xlua C90_sys, X-Plane 12.4.3",
        live=(switches.C90_SIGNS.position,),
    ),
    Override(
        name="b738_switches",
        requires=(
            switches.B738_POSITION.down,
            switches.B738_POSITION.up,
            switches.B738_SEATBELT.down,
            switches.B738_SEATBELT.up,
        ),
        commands=MappingProxyType({}),
        # In the air its speed brake lever stops at the flight detent.
        settings=MappingProxyType({"switches": switches.B738, "speed_brake_air_max": "half"}),
        source="Laminar 737-800 b738_cockpit.obj manipulators and xlua B738.lighting / "
        "B738.systems, X-Plane 12.4.3",
        live=(switches.B738_POSITION.position,),
    ),
    Override(
        name="a333_switches",
        requires=(
            switches.A333_NAV.down,
            switches.A333_NAV.up,
            switches.A333_SEATBELT.down,
            switches.A333_SEATBELT.up,
        ),
        commands=MappingProxyType({}),
        settings=MappingProxyType(
            {"switches": switches.A333, "autobrake_off": switches.A333_AUTOBRAKE_OFF}
        ),
        source="Laminar A330-300 A330_cockpit.obj manipulators and xlua A333.lighting / "
        "A333.switches, X-Plane 12.4.3",
        live=(switches.A333_STROBE,),
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
    # Whether the aircraft's own files (.acf, cockpit objects, scripts) use an
    # autothrottle at all; None until scanned. X-Plane's generic autothrottle
    # runs on any aircraft (the Baron and King Air probes said "works"), so
    # this, not the probe, says whether the aircraft has one.
    autothrottle_shown: bool | None = None

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
    settings: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    # Roles whose command is the aircraft's own (from an override).
    overridden: frozenset[str] = frozenset()

    def flap_notch(self, label: int) -> int | None:
        """The handle notch for a spoken flap setting ("flaps fifteen"), or None
        when this aircraft has no such setting.

        1. Labels from an override (737: 1 2 5 10 15 25 30 40), used only when
           their count matches the aircraft's detents.
        2. Else a notch whose modelled angle is within 3 degrees (Baron 13 and
           30: "flaps fifteen" -> notch 1).
        3. Else the notch number itself, if there is one ("flaps one", "flaps two").
        """
        detents = self.info.flap_detents
        if not detents:
            return None
        labels = self.settings.get("flap_labels") or ()
        full_word = 1 if len(labels) == detents - 1 else 0  # A330: 1 2 3 + "full"
        if labels and len(labels) + full_word == detents:
            return labels.index(str(label)) + 1 if str(label) in labels else None
        angles = self.info.flap_notch_degrees()
        near = [i for i, a in enumerate(angles) if i and abs(a - label) <= 3.0]
        if near:
            return min(near, key=lambda i: abs(angles[i] - label))
        return label if 1 <= label <= detents else None

    @property
    def engine_start(self) -> str:
        kind = self.info.engine_kind
        return kind if kind in ("piston", "jet", "turboprop", "electric") else "piston"

    def command(self, role: str) -> str | None:
        return self.commands.get(role)

    def supports(self, role: str) -> bool | None:
        """True / False when measured, None when nobody has probed it.

        A "no change" on the aircraft's *own* control (an override) is not a
        refusal: the control exists and did nothing in that moment. The 737's
        LNAV/VNAV buttons need a flight plan, and the A330's A/THR held while
        the levers were at FLX. Those are tried and verified (None)."""
        verdict = self.probed.get(role)
        if verdict is None:
            return None if self.command(role) else False
        if verdict == "no_change" and role in self.overridden:
            return None
        return verdict == "works"

    def switch_plan(self, key: str, setting: str) -> tuple[str, switches.Plan]:
        """How to reach ``setting`` of switch ``key`` here (switches.lookup).
        Without an aircraft's own autobrake entry, only jets get X-Plane's."""
        own = self.settings.get("switches") or {}
        if key == "autobrake" and key not in own and self.info.engine_kind != "jet":
            return "absent", ()
        return switches.lookup(own, key, setting)

    def status_dataref(self, role: str, default: str) -> str:
        """Where this aircraft reports a mode's state (A330 VNAV: fms_vnav)."""
        return (self.settings.get("status_datarefs") or {}).get(role, default)

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
    has_dataref: Callable[[str], bool] = lambda _name: True,
) -> Profile:
    """Layers 1-4 -> the profile the plugin uses for this aircraft.
    ``has_dataref``: is a dataref registered now (for ``Override.live``)."""
    commands: dict[str, str | None] = dict(DEFAULT_COMMANDS)
    settings: dict[str, Any] = {}
    applied = []
    overridden: set[str] = set()
    for override in OVERRIDES:
        if all(has_command(name) for name in override.requires) and all(
            has_dataref(name) for name in override.live
        ):
            commands.update(override.commands)
            settings.update(override.settings)
            applied.append(override.name)
            overridden |= {role for role, name in override.commands.items() if name}
    return Profile(
        info=info,
        commands=MappingProxyType(commands),
        overrides=tuple(applied),
        probed=MappingProxyType(dict(probed or {})),
        settings=MappingProxyType(settings),
        overridden=frozenset(overridden),
    )


# What an aircraft's files say when it has an autothrottle: its panel, 3D
# cockpit or scripts use the sim's autothrottle datarefs or commands.
AUTOTHROTTLE_NEEDLE = b"autothrottle"
_SCAN_SUFFIXES = (".acf", ".lua", ".obj")
_SCAN_MAX_BYTES = 600_000_000


def files_mention(acf_path: str, needle: bytes = AUTOTHROTTLE_NEEDLE) -> bool | None:
    """Whether the aircraft's .acf, scripts or objects mention ``needle``
    (case-insensitive). The .acf first, then scripts, then objects, stopping
    at the first hit; liveries are skipped. None when the folder can't be
    read or the scan would be too large to be sure.

    Checked on X-Plane 12.4.3: the 737 (in its .acf), A330 (.acf) and MD-82
    (cockpit objects, scripts) mention it; the Baron, King Air C90B, C172,
    SR22 and Citation X mention it nowhere."""
    folder = os.path.dirname(acf_path)
    if not acf_path or not os.path.isdir(folder):
        return None
    files: list[str] = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if d.lower() != "liveries"]
        files += [os.path.join(root, n) for n in names if n.lower().endswith(_SCAN_SUFFIXES)]
    order = {".acf": 0, ".lua": 1, ".obj": 2}
    files.sort(key=lambda f: order[os.path.splitext(f)[1].lower()])
    needle = needle.lower()
    total = 0
    try:
        for f in files:
            total += os.path.getsize(f)
            if total > _SCAN_MAX_BYTES:
                return None
            with open(f, "rb") as fh:
                if needle in fh.read().lower():
                    return True
    except OSError:
        return None
    return False


_FLAP_LIMIT = re.compile(rb"^P acf/(_Vfe1_kts|_Vfem_kts) ([0-9.]+)\s*$", re.M)


def acf_flap_limits(acf_path: str) -> dict[str, float]:
    """The flap speed limits in an aircraft's .acf, KIAS: ``first`` for the
    first notch (``_Vfe1_kts``, which X-Plane publishes as acf_Vfe) and
    ``full`` for full flaps (``_Vfem_kts``, which it doesn't publish).

    Read on X-Plane 12.4.3 (2026-11-12): Baron 58 152 / 122, King Air C90B
    190 / 150, 737-800 250 / 162, A330-300 215 / 180. Using acf_Vfe as the
    full-flap limit let the 737 take full flaps at 240 kt."""
    try:
        with open(acf_path, "rb") as fh:
            data = fh.read()
    except OSError:
        return {}
    names = {b"_Vfe1_kts": "first", b"_Vfem_kts": "full"}
    return {names[m.group(1)]: float(m.group(2)) for m in _FLAP_LIMIT.finditer(data)}


def flap_limit_kt(
    notch: int, detents: int, first: float | None, full: float | None
) -> float | None:
    """The speed limit for flap notch ``notch`` of ``detents``: the first
    notch's limit at notch 1, the full-flap limit at the last, evenly in
    between (only those two are published). On the 737 that gives 250, 237,
    225, 212, 200, 187, 175, 162 kt for flaps 1 to 40, within a few knots of
    its placard (250, 250, 250, 210, 200, 190, 175, 162)."""
    if notch <= 0 or detents <= 0:
        return None
    if full is None or first is None:
        return full if first is None else first
    if detents == 1 or notch >= detents:
        return full
    return round(first + (full - first) * (notch - 1) / (detents - 1), 1)


def override_names() -> tuple[str, ...]:
    """Every aircraft-specific command and dataref the overrides use
    (reference checker)."""
    names = {n for o in OVERRIDES for n in (*o.requires, *o.live)}
    names |= {v for o in OVERRIDES for v in o.commands.values() if v}
    for o in OVERRIDES:
        for knobs in (o.settings.get("xpdr_knobs") or {}).values():
            names |= {n for k in knobs for n in (k.position, k.down, k.up)}
        # The aircraft's own switches; X-Plane's sim/... ones are checked
        # against its lists with the rest of the source.
        own = switches.plan_names(o.settings.get("switches") or {})
        names |= {re.sub(r"\[\d+\]$", "", n) for n in own if not n.startswith("sim/")}
    return tuple(sorted(names))
