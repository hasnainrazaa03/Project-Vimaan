"""The aircraft probe: measure which controls work on the loaded aircraft.

Manuals don't say whether an aircraft honours X-Plane's generic commands
beyond its own buttons (does the Baron, which has no V/S button, accept
``sim/autopilot/vertical_speed``?). The probe finds out on the aircraft, for
*any* aircraft, and saves the answer per aircraft file. That file is layer 4 of
``vimaan_nlu.aircraft``: Vimaan says "not on this aircraft" for a measured
"no change" instead of pressing it and hoping.

The pilot runs it: one key press per step, with the aircraft stable. A step
snapshots the autopilot, presses one control or writes one value, reads back
after a pause, classifies the result, and restores the previous state by
pressing the control again and, if a mode is still different, the control for
the mode that was on before. Mode datarefs can't be written without an
autopilot override, so restoring is done with buttons. If restoring fails,
the probe stops and says so.

Every measured step is saved at once, with the place the run has reached, so
a stopped run (or X-Plane quitting) loses nothing and the next press carries on
from there.

Pure Python: the plugin does the reading, pressing and timing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SETTLE_SEC = 2.0  # how long a step waits before reading back
AIR_MIN_AGL_FT = 1500.0

# State the probe reads: key -> (dataref, is_float).
STATE: Mapping[str, tuple[str, bool]] = {
    "servos_on": ("sim/cockpit2/autopilot/servos_on", False),
    "lateral_mode": ("sim/cockpit2/autopilot/heading_mode", False),
    "vertical_mode": ("sim/cockpit2/autopilot/altitude_mode", False),
    "heading_status": ("sim/cockpit2/autopilot/heading_status", False),
    "nav_status": ("sim/cockpit2/autopilot/nav_status", False),
    "approach_status": ("sim/cockpit2/autopilot/approach_status", False),
    "backcourse_status": ("sim/cockpit2/autopilot/backcourse_status", False),
    "gpss_status": ("sim/cockpit2/autopilot/gpss_status", False),
    "altitude_hold_status": ("sim/cockpit2/autopilot/altitude_hold_status", False),
    "altitude_armed": ("sim/cockpit2/autopilot/altitude_hold_armed", False),
    "vs_status": ("sim/cockpit2/autopilot/vvi_status", False),
    "speed_status": ("sim/cockpit2/autopilot/speed_status", False),
    "vnav_status": ("sim/cockpit2/autopilot/vnav_status", False),
    "autothrottle_enabled": ("sim/cockpit2/autopilot/autothrottle_enabled", False),
    "autothrottle_on": ("sim/cockpit2/autopilot/autothrottle_on", False),
    "airspeed_is_mach": ("sim/cockpit2/autopilot/airspeed_is_mach", False),
    "speed_dial": ("sim/cockpit2/autopilot/airspeed_dial_kts_mach", True),
    "vs_dial": ("sim/cockpit2/autopilot/vvi_dial_fpm", True),
    "altitude_dial": ("sim/cockpit2/autopilot/altitude_dial_ft", True),
    "baro": ("sim/cockpit2/gauges/actuators/barometer_setting_in_hg_pilot", True),
    "flap_handle": ("sim/cockpit2/controls/flap_handle_request_ratio", True),
}
# What must be back as it was after a step (modes, armed states, A/T, units).
RESTORE_KEYS = (
    "servos_on",
    "lateral_mode",
    "vertical_mode",
    "nav_status",
    "approach_status",
    "backcourse_status",
    "altitude_armed",
    "autothrottle_enabled",
    "airspeed_is_mach",
)
# The control that selects a mode, for putting the previous mode back.
LATERAL_ROLES = {1: "mode_heading", 2: "mode_nav", 13: "mode_gpss"}
VERTICAL_ROLES = {
    4: "mode_vertical_speed",
    5: "mode_level_change",
    6: "mode_altitude_hold",
    9: "mode_vnav",
}
STANDARD_BARO_INHG = 29.92


@dataclass(frozen=True)
class Step:
    role: str  # a vimaan_nlu.aircraft role; also the result key
    label: str  # spoken
    where: str  # "air" | "ground"
    watch: tuple[str, ...] = ()  # STATE keys a working control changes
    write: str | None = None  # STATE key to write instead of pressing a command
    delta: float = 0.0  # written value = current + delta
    tolerance: float = 0.5  # a write "held" within this


STEPS: tuple[Step, ...] = (
    Step("mode_heading", "heading mode", "air", ("heading_status", "lateral_mode")),
    Step("mode_nav", "nav mode", "air", ("nav_status", "lateral_mode")),
    Step("mode_approach", "approach mode", "air", ("approach_status",)),
    Step("mode_back_course", "back course", "air", ("backcourse_status",)),
    Step("mode_gpss", "GPS steering", "air", ("gpss_status", "lateral_mode")),
    Step("mode_altitude_hold", "altitude hold", "air", ("altitude_hold_status", "vertical_mode")),
    Step("mode_altitude_arm", "altitude arm", "air", ("altitude_armed",)),
    Step("mode_vertical_speed", "vertical speed mode", "air", ("vs_status", "vertical_mode")),
    Step("mode_level_change", "level change", "air", ("speed_status", "vertical_mode")),
    Step("mode_vnav", "V NAV", "air", ("vnav_status", "vertical_mode")),
    Step("autothrottle_toggle", "autothrottle", "air", ("autothrottle_enabled", "autothrottle_on")),
    Step("knots_mach_toggle", "knots and Mach", "air", ("airspeed_is_mach",)),
    Step("dial_speed", "speed dial", "air", write="speed_dial", delta=5.0),
    Step("dial_vertical_speed", "vertical speed dial", "air", write="vs_dial", delta=100.0),
    Step("dial_altitude", "altitude dial", "air", write="altitude_dial", delta=100.0),
    Step("baro_std", "altimeter standard", "ground", ("baro",)),
    Step("dial_baro", "altimeter setting", "ground", write="baro", delta=0.10, tolerance=0.005),
    # Written as one notch of travel at runtime (1 / detents).
    Step("flap_handle", "flap handle", "ground", write="flap_handle", tolerance=0.01),
)


def steps_for(where: str) -> tuple[Step, ...]:
    return tuple(s for s in STEPS if s.where == where)


def situation(
    on_ground: bool | None, agl_ft: float | None, ground_speed_kt: float | None
) -> str | None:
    """Which set of steps the aircraft is in a state to run: "ground" when
    stopped on the ground, "air" when airborne at a safe height, else None."""
    if on_ground and ground_speed_kt is not None and ground_speed_kt < 1.0:
        return "ground"
    if on_ground is False and agl_ft is not None and agl_ft >= AIR_MIN_AGL_FT:
        return "air"
    return None


def ready_for(step: Step, state: Mapping[str, Any], parking_brake: bool | None) -> str | None:
    """Why ``step`` can't run now, or None when it can."""
    if step.where == "air" and state.get("servos_on") != 1:
        return "Turn the autopilot on first"
    if step.where == "ground" and not parking_brake:
        return "Set the parking brake first"
    return None


def caution(step: Step, state: Mapping[str, Any]) -> str | None:
    """A reason the step may not put things back, which the pilot can choose
    to ignore (one more press). Restoring presses the mode that was on before;
    roll and pitch hold have no button, so with them on nothing can be pressed.
    In flight, the 737 engaged with no modes and its ALT HOLD step failed."""
    if step.where != "air":
        return None
    if (
        state.get("lateral_mode") not in LATERAL_ROLES
        or state.get("vertical_mode") not in VERTICAL_ROLES
    ):
        return "Heading and altitude hold are not both on, so this step may not restore"
    return None


def write_value(step: Step, state: Mapping[str, Any], flap_detents: int | None) -> float | None:
    if step.write is None:
        return None
    if step.write == "flap_handle":
        return 1.0 / flap_detents if flap_detents else None
    current = state.get(step.write)
    return None if current is None else float(current) + step.delta


# How close a float must come back to count as restored: an aircraft's own
# logic rounds (the 737 put the altimeter back at 29.92 for 29.9212).
TOLERANCE: Mapping[str, float] = {
    "baro": 0.006,  # inHg
    "flap_handle": 0.01,
    "speed_dial": 0.5,  # knots; Mach is checked at 0.005 below
    "vs_dial": 10.0,
    "altitude_dial": 5.0,
}


def _same(key: str, a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a == b
    if STATE[key][1]:
        tolerance = TOLERANCE.get(key, 1e-3)
        if key == "speed_dial" and max(abs(float(a)), abs(float(b))) < 2:
            tolerance = 0.005  # the dial holds a Mach number
        return abs(float(a) - float(b)) <= tolerance
    return a == b


def autopilot_dropped(step: Step, before: Mapping[str, Any], now: Mapping[str, Any]) -> bool:
    """The autopilot was on for an air step and isn't now: the aircraft (or
    the pilot's controls) disconnected it, which says nothing about the
    control being tested."""
    return step.where == "air" and before.get("servos_on") == 1 and now.get("servos_on") == 0


def classify(
    step: Step,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    found: bool,
    written: float | None = None,
) -> str:
    """works / no_change / missing / inconclusive."""
    if step.write is not None:
        if written is None:
            return "inconclusive"
        held = after.get(step.write)
        return (
            "works" if held is not None and abs(held - written) <= step.tolerance else "no_change"
        )
    if not found:
        return "missing"
    if step.role == "baro_std":
        was_std = before.get("baro") is not None and abs(before["baro"] - STANDARD_BARO_INHG) < 0.01
        now_std = after.get("baro") is not None and abs(after["baro"] - STANDARD_BARO_INHG) < 0.01
        if was_std:
            return "inconclusive"  # already standard: pressing it shows nothing
        return "works" if now_std else "no_change"
    changed = any(not _same(k, before.get(k), after.get(k)) for k in step.watch)
    return "works" if changed else "no_change"


def _restore_keys(step: Step) -> tuple[str, ...]:
    return (
        RESTORE_KEYS
        + ((step.write,) if step.write else ())
        + (("baro",) if step.role == "baro_std" else ())
    )


def leftovers(step: Step, before: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    """What hasn't gone back to how it was (STATE keys)."""
    return [k for k in _restore_keys(step) if not _same(k, before.get(k), now.get(k))]


def restored(step: Step, before: Mapping[str, Any], now: Mapping[str, Any]) -> bool:
    return not leftovers(step, before, now)


# Left over on its own, this doesn't stop the probe. On the 737, moving the
# altitude dial arms altitude capture, and it stays armed after the dial goes
# back: armed for the altitude the aircraft is already at, it changes nothing.
HARMLESS_LEFTOVERS = frozenset({"altitude_armed"})

LEFTOVER_NAMES: Mapping[str, str] = {
    "servos_on": "the autopilot",
    "lateral_mode": "the lateral mode",
    "vertical_mode": "the vertical mode",
    "nav_status": "nav",
    "approach_status": "approach",
    "backcourse_status": "back course",
    "altitude_armed": "altitude arm",
    "autothrottle_enabled": "the autothrottle",
    "airspeed_is_mach": "knots or Mach",
    "speed_dial": "the speed dial",
    "vs_dial": "the vertical speed dial",
    "altitude_dial": "the altitude dial",
    "baro": "the altimeter",
    "flap_handle": "the flap handle",
}


def describe(keys: list[str]) -> str:
    """A spoken list of STATE keys: "nav and approach"."""
    names = [LEFTOVER_NAMES.get(k, k) for k in keys]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def restore_presses(
    before: Mapping[str, Any], now: Mapping[str, Any], attempt: int
) -> list[str] | None:
    """Roles to press on restore attempt ``attempt``, or None when there are no
    more attempts. First the previous modes; then, once the modes have
    settled, an altitude arm or autothrottle the step left changed (the 737's
    LVL CHG turned the autothrottle on). Arm and A/T wait for the modes
    because pressing a mode can clear an arm by itself."""
    if attempt == 0:
        return mode_fix(before, now)
    if attempt == 1:
        roles = []
        if bool(now.get("altitude_armed")) != bool(before.get("altitude_armed")):
            roles.append("mode_altitude_arm")
        if autothrottle_active(now) != autothrottle_active(before):
            roles.append("autothrottle_toggle")
        return roles
    return None


def autothrottle_active(state: Mapping[str, Any]) -> bool:
    """X-Plane's autothrottle_enabled: -1 hard off, 0 armed, 1-3 holding
    speed / N1 / retard. The A330's A/THR button flips -1 and 0."""
    value = state.get("autothrottle_enabled")
    return value is not None and value > 0


def undo_needed(step: Step, before: Mapping[str, Any], now: Mapping[str, Any]) -> bool:
    """After the autopilot dropped mid-step: did the pressed control change
    what it controls? Then press it again before pausing, so the retry (and
    the steps after it) start from where the run was. In flight the A330's
    APP stayed armed after a pause, and the next step couldn't restore."""
    return any(not _same(k, before.get(k), now.get(k)) for k in step.watch)


def mode_fix(before: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    """Controls that put the previous lateral/vertical mode back."""
    roles = []
    if now.get("lateral_mode") != before.get("lateral_mode"):
        role = LATERAL_ROLES.get(before.get("lateral_mode"))  # type: ignore[arg-type]
        if role:
            roles.append(role)
    if now.get("vertical_mode") != before.get("vertical_mode"):
        role = VERTICAL_ROLES.get(before.get("vertical_mode"))  # type: ignore[arg-type]
        if role:
            roles.append(role)
    return roles


def changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, list[Any]]:
    return {
        k: [before.get(k), after.get(k)] for k in STATE if not _same(k, before.get(k), after.get(k))
    }


# ---------------------------------------------------------------------------
# Results file (one per aircraft file)
# ---------------------------------------------------------------------------
VERDICTS = ("works", "no_change", "missing")


def results_path(folder: str, key: str) -> str:
    return os.path.join(folder, f"{key}.json")


def load_record(path: str) -> dict[str, Any]:
    """The saved probe file, or {} when there is none (or it is broken)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_results(path: str) -> dict[str, str]:
    """role -> verdict, from a saved probe (empty when there is none). Only
    definite verdicts count; "inconclusive" and skipped steps are left out."""
    steps = load_record(path).get("steps", {})
    return {k: v for k, v in steps.items() if v in VERDICTS}


def resume_index(record: Mapping[str, Any], where: str) -> int:
    """Where an unfinished run of ``where`` steps stopped: the index of the
    step to run next, 0 for a fresh start. Saved as a role, so a change to the
    step list can't point it at the wrong step."""
    role = (record.get("resume") or {}).get(where)
    roles = [s.role for s in steps_for(where)]
    return roles.index(role) if role in roles else 0


def counts(record: Mapping[str, Any], where: str) -> dict[str, int]:
    """Verdict -> how many ``where`` steps have it, over every saved run."""
    steps = record.get("steps", {})
    out: dict[str, int] = {}
    for s in steps_for(where):
        verdict = steps.get(s.role)
        if verdict:
            out[verdict] = out.get(verdict, 0) + 1
    return out


def save_results(path: str, record: Mapping[str, Any]) -> None:
    """Merge with what is saved for the aircraft, then write atomically.

    Called after every step and again when a run ends. Steps and the resume
    point merge per key (a ground run and an air run each fill in their own);
    details append. A new "inconclusive" doesn't replace a definite verdict
    from an earlier run: the 737's STD step is inconclusive whenever the
    altimeter is already at 29.92. A resume point of None clears it."""
    previous = load_record(path)
    merged = {**previous, **record}
    steps = dict(previous.get("steps", {}))
    for role, verdict in record.get("steps", {}).items():
        if verdict in VERDICTS or steps.get(role) not in VERDICTS:
            steps[role] = verdict
    merged["steps"] = steps
    merged["details"] = [*previous.get("details", []), *record.get("details", [])][-200:]
    resume = {**previous.get("resume", {}), **record.get("resume", {})}
    merged["resume"] = {k: v for k, v in resume.items() if v}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
