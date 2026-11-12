"""Flights: each X-Plane session's Vimaan log as something to review.

The plugin writes ``vimaan_<YYYYMMDD_HHMMSS>.events.jsonl`` per session (see
``ML/vimaan_nlu/telemetry.py``) and a recording per command under
``audio/<session>_<cmd>.wav``. This module turns a session into:

- one row per push-to-talk command: what was heard and understood, what
  Vimaan did to the sim and said back, the outcome, how long it took;
- the autopilot changes, and whether Vimaan made them;
- startup, the aircraft and joystick, the probe, and anything that went wrong.

Read-only, stdlib only. A session still being written (a flight in progress)
is read the same way: a torn last line is skipped until it is complete.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import paths

SESSION_RE = re.compile(r"^\d{8}_\d{6}$")
CMD_RE = re.compile(r"^c\d{4}$")
# A session with no end record whose log changed this recently is in progress
# (the plugin writes a heartbeat every minute).
LIVE_SEC = 150.0
# An autopilot change this soon after "autopilot ..." from Vimaan is Vimaan's.
VIMAAN_SEC = 5.0

# command_done outcome -> (group, label). Groups drive the filters and the
# status colour; the label is always shown with it.
OUTCOMES: dict[str, tuple[str, str]] = {
    "executed": ("done", "Done"),
    "no_op": ("done", "Already set"),
    "answered": ("done", "Answered"),
    "pending": ("asked", "Asked"),
    "safety_hold": ("asked", "Asked to confirm"),
    "ambiguous": ("asked", "Asked which"),
    "cancelled": ("asked", "Cancelled"),
    "rejected": ("refused", "Not done"),  # Vimaan says why ("On or off?")
    "non_actionable": ("talk", "Not a command"),
    "low_confidence": ("missed", "Unsure, ignored"),
    "not_understood": ("missed", "Not understood"),
    "too_short": ("missed", "Too short"),
    "silent_mic": ("missed", "Mic silent"),
    "failed": ("failed", "Failed"),
    "timeout": ("failed", "Timed out"),
    "worker_error": ("failed", "Error"),
    "mic_error": ("failed", "Mic error"),
    "stt_service_error": ("failed", "Recognition error"),
}
GROUPS = ("done", "asked", "refused", "talk", "missed", "failed", "other")

# Events that mean something went wrong (as in scripts/vimaan_log_report.py).
PROBLEM_KINDS = {
    "dependency_missing",
    "microphone_unavailable",
    "stage_failed",
    "uncaught_exception",
    "execute_error",
    "listen_worker_error",
    "flight_loop_error",
    "sim_stall",
    "dataref_read_error",
    "dataref_write_error",
    "worker_spawn_failed",
    "worker_exit",
    "worker_fatal",
    "stt_fallback",
    "overlay_failed",
}

# Autopilot state (the plugin's AUTOPILOT_STATE) in words.
_ONOFF = {0: "off", 1: "on"}
_STATUS = {0: "off", 1: "armed", 2: "on"}
_LATERAL = {
    0: "roll",
    1: "heading",
    2: "nav",
    10: "TO/GA",
    11: "re-entry",
    12: "free",
    13: "GPS steering",
    14: "heading hold",
    15: "turn rate",
    16: "rollout",
    18: "track",
}
_VERTICAL = {
    3: "pitch",
    4: "vertical speed",
    5: "level change",
    6: "altitude hold",
    7: "terrain follow",
    8: "glideslope",
    9: "VNAV path",
    10: "TO/GA",
    11: "re-entry",
    12: "free",
    17: "flare",
    19: "flight path",
}
AUTOPILOT_FIELDS: dict[str, tuple[str, dict[int, str]]] = {
    "servos_on": ("Autopilot", _ONOFF),
    "flight_director": ("Flight director", {0: "off", 1: "on", 2: "on"}),
    "lateral_mode": ("Lateral mode", _LATERAL),
    "vertical_mode": ("Vertical mode", _VERTICAL),
    "heading_status": ("Heading", _STATUS),
    "nav_status": ("Nav", _STATUS),
    "approach_status": ("Approach", _STATUS),
    "altitude_hold_status": ("Altitude hold", _STATUS),
    "altitude_armed": ("Altitude capture", {0: "off", 1: "armed", 2: "armed"}),
    "vs_status": ("Vertical speed", _STATUS),
    "speed_status": ("Speed", _STATUS),
}

_cache: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
_display: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def log_dir() -> Path:
    return paths.FLIGHT_LOGS


def events_path(session: str) -> Path | None:
    if not isinstance(session, str) or not SESSION_RE.match(session):
        return None
    p = log_dir() / f"vimaan_{session}.events.jsonl"
    return p if p.is_file() else None


def audio_path(session: str, cmd: str) -> Path | None:
    if not SESSION_RE.match(session or "") or not CMD_RE.match(cmd or ""):
        return None
    p = log_dir() / "audio" / f"{session}_{cmd}.wav"
    return p if p.is_file() else None


def _sessions() -> list[str]:
    root = log_dir()
    if not root.is_dir():
        return []
    out = []
    for p in root.glob("vimaan_*.events.jsonl"):
        session = p.name[len("vimaan_") : -len(".events.jsonl")]
        if SESSION_RE.match(session):
            out.append(session)
    return sorted(out, reverse=True)


def _read(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line still being written, or a crash mid-write
    return rows


def _intent_names() -> dict[str, str]:
    """Display names from the command table (pure Python, in ML/)."""
    global _display
    if _display is None:
        try:
            if str(paths.ML_DIR) not in sys.path:
                sys.path.insert(0, str(paths.ML_DIR))
            from vimaan_nlu.commands import COMMANDS

            _display = {c.intent: c.display for c in COMMANDS}
        except Exception:
            _display = {}
    return _display


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------
def _clock(ev: dict[str, Any]) -> str | None:
    ts = ev.get("ts")
    return ts[11:19] if isinstance(ts, str) and len(ts) >= 19 else None


def _new_command(cmd: str, ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "command",
        "cmd": cmd,
        "t": ev.get("t"),
        "clock": _clock(ev),
        "source": ev.get("source"),
        "held_ms": None,
        "audio": False,
        "audio_sec": None,
        "heard": None,
        "stt_ms": None,
        "normalized": None,
        "intent": None,
        "intent_name": None,
        "confidence": None,
        "slots": {},
        "outcome": None,
        "group": None,
        "label": None,
        "replies": [],
        "actions": [],
        "checks": [],
        "notes": [],
        "latency_ms": None,
        "total_ms": None,
        "slow": False,
    }


def _note(c: dict[str, Any], kind: str, text: str, tone: str = "info") -> None:
    c["notes"].append({"kind": kind, "text": text, "tone": tone})


def _command_event(c: dict[str, Any], ev: dict[str, Any], names: dict[str, str]) -> None:
    """Fold one event that carries this command's id into its row."""
    kind = ev.get("kind")
    if kind == "ptt_release":
        c["held_ms"] = ev.get("held_ms")
    elif kind == "stage":
        stage = ev.get("stage")
        if stage == "capture":
            c["audio_sec"] = ev.get("audio_sec")
            c["audio"] = bool(ev.get("audio_file"))
        elif stage == "stt":
            c["heard"], c["stt_ms"] = ev.get("text"), ev.get("ms")
        elif stage == "nlu":
            c["normalized"] = ev.get("normalized")
            c["intent"] = ev.get("intent")
            c["intent_name"] = names.get(ev.get("intent") or "", ev.get("intent"))
            c["confidence"] = ev.get("confidence")
            c["slots"] = ev.get("slots") or {}
        elif stage == "dispatch" and c["outcome"] is None:
            c["outcome"] = ev.get("outcome")
    elif kind == "speak":
        if ev.get("text"):
            c["replies"].append(ev["text"])
    elif kind == "sim_command":
        c["actions"].append({"kind": "command", "name": ev.get("command"), "ok": ev.get("ok")})
    elif kind == "sim_write":
        c["actions"].append(
            {
                "kind": "write",
                "name": ev.get("dataref"),
                "value": ev.get("value"),
                "readback": ev.get("readback"),
                "ok": ev.get("ok"),
            }
        )
    elif kind == "verify":
        c["checks"].append(
            {
                "label": ev.get("label"),
                "expected": ev.get("expected"),
                "actual": ev.get("actual"),
                "ok": ev.get("ok"),
            }
        )
    elif kind == "command_done":
        c["outcome"] = ev.get("outcome") or c["outcome"]
        c["latency_ms"] = ev.get("speech_to_action_ms")
        c["total_ms"] = ev.get("total_ms")
        c["slow"] = bool(ev.get("slow"))
        if not c["heard"] and ev.get("text"):
            c["heard"] = ev.get("text")
    elif kind == "worker_error":
        _note(c, kind, f"Not heard: {ev.get('outcome') or ev.get('error')}", "warn")
    elif kind == "stage_ended":
        _note(c, kind, f"Recognition ended: {ev.get('reason')}", "warn")
    elif kind == "safety_hold":
        _note(c, kind, f"Held by safety rule {ev.get('rule')}", "warn")
    elif kind == "safety_override":
        _note(c, kind, f"Override of {ev.get('rule')} for {ev.get('held_cmd')}", "warn")
    elif kind == "override_with_command":
        _note(c, kind, "Override said with the command")
    elif kind == "override_ignored":
        _note(c, kind, f"Nothing to override ({ev.get('reason')})")
    elif kind == "slot_rejected":
        _note(c, kind, f"{ev.get('slot')} {ev.get('value')!r} refused: {ev.get('reason')}", "warn")
    elif kind == "implausible":
        _note(c, kind, f"Ignored as implausible: {ev.get('reason')}", "warn")
    elif kind == "ambiguous":
        _note(c, kind, f"'to' or 'two'? {ev.get('as_to')} or {ev.get('as_two')}")
    elif kind == "bare_command":
        _note(
            c, kind, f"Just the name: asked about {names.get(ev.get('intent'), ev.get('intent'))}"
        )
    elif kind == "answered":
        _note(c, kind, f"Answered the question from {ev.get('reply_to')}")
    elif kind == "cancelled":
        _note(c, kind, "Cancelled")
    elif kind == "not_applicable":
        _note(c, kind, f"Not for this aircraft: {ev.get('reason')}")
    elif kind in ("capture_start", "capture_end", "capture_setting"):
        _note(c, kind, _capture_text(ev))
    elif kind in ("baro_step_start", "baro_step_end"):
        _note(c, kind, "Altimeter knob " + ("stepping" if kind.endswith("start") else "done"))
    elif kind == "crank_end":
        tone = "info" if ev.get("result") == "running" else "warn"
        _note(
            c, kind, f"{ev.get('engine')}: {ev.get('result')} after {ev.get('crank_sec')} s", tone
        )
    elif kind == "aircraft_state" and ev.get("after_sec"):
        _note(
            c,
            kind,
            f"+{ev.get('after_sec'):.0f} s: alt {ev.get('altitude_actual')} "
            f"(target {ev.get('altitude_target')}), hdg {ev.get('heading_actual')} "
            f"(target {ev.get('heading_target')}), vs {ev.get('vs_actual')}",
        )


def _capture_text(ev: dict[str, Any]) -> str:
    kind = ev.get("kind")
    if kind == "capture_start":
        return f"Vimaan capture to {ev.get('target')} ft at {ev.get('fpm')} ft/min"
    if kind == "capture_end":
        return f"Vimaan capture ended: {ev.get('reason') or ev.get('result')}"
    return f"Vimaan capture {'on' if ev.get('enabled') else 'off'}"


def _autopilot_changes(changed: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key, (name, words) in AUTOPILOT_FIELDS.items():
        if key not in changed:
            continue
        a, b = changed[key]
        out.append({"name": name, "from": words.get(a, a), "to": words.get(b, b)})
    return out


def _went_off(changed: dict[str, Any]) -> bool:
    pair = changed.get("servos_on") or changed.get("autopilot_on")
    return bool(pair) and pair[0] == 1 and pair[1] == 0


def _event(ev: dict[str, Any], text: str, tone: str = "info", **extra: Any) -> dict[str, Any]:
    return {
        "type": "event",
        "kind": ev.get("kind"),
        "t": ev.get("t"),
        "clock": _clock(ev),
        "text": text,
        "tone": tone,
        **extra,
    }


def _aircraft_name(ev: dict[str, Any]) -> str:
    acf = ev.get("acf") or ""
    folder = Path(acf).parent.name if acf else ""
    return folder or ev.get("icao") or "aircraft"


def parse(rows: list[dict[str, Any]], *, live: bool = False) -> dict[str, Any]:
    """A session's events -> {summary, items, problems}."""
    names = _intent_names()
    commands: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    aircraft: list[str] = []
    model = stt = None
    self_test: dict[str, Any] | None = None
    ended = False
    probe_on = False
    drops_seen_by_plugin = any(r.get("kind") == "autopilot_disconnect" for r in rows)
    ap_drops = 0
    joystick: dict[str, Any] | None = None

    for ev in rows:
        kind = ev.get("kind")
        cmd = ev.get("cmd")
        if kind == "ptt_press" and isinstance(cmd, str):
            commands[cmd] = _new_command(cmd, ev)
            items.append(commands[cmd])
            continue
        if isinstance(cmd, str) and cmd in commands:
            _command_event(commands[cmd], ev, names)
            if kind not in PROBLEM_KINDS:
                continue

        if kind == "session_start":
            stt = ev.get("stt_backend")
            items.append(_event(ev, f"Session started ({ev.get('backend')}, {stt} speech)"))
        elif kind == "stage" and ev.get("stage") == "model_load":
            model = Path(str(ev.get("model_path") or "")).name or None
            secs = (ev.get("ms") or 0) / 1000
            items.append(_event(ev, f"Model {model or '?'} loaded ({secs:.1f} s)"))
        elif kind == "worker_ready":
            items.append(_event(ev, f"Voice engine ready ({ev.get('boot_sec')} s)"))
        elif kind == "stage" and ev.get("stage") == "self_test":
            self_test = {"passed": ev.get("passed"), "total": ev.get("total")}
            ok = ev.get("passed") == ev.get("total")
            items.append(
                _event(
                    ev,
                    f"Self test {ev.get('passed')} of {ev.get('total')} passed",
                    "good" if ok else "bad",
                )
            )
        elif kind == "aircraft":
            name = _aircraft_name(ev)
            if name not in aircraft:
                aircraft.append(name)
            items.append(_event(ev, f"Aircraft: {name} ({ev.get('icao')})"))
        elif kind == "joystick":
            joystick = {k: ev.get(k) for k in ("has_joystick", "nullzone", "stick")}
            nz = ev.get("nullzone")
            if ev.get("has_joystick") and nz is not None and nz < 0.02:
                items.append(
                    _event(
                        ev,
                        f"Joystick null zone is {nz * 100:.0f}%: a resting stick can fly "
                        "the aircraft and take the autopilot off. 10-15% is usual.",
                        "warn",
                    )
                )
        elif kind == "autopilot_state_change":
            changed = ev.get("changed") or {}
            sec = ev.get("sec_since_vimaan_action")
            vimaan = (
                ev.get("last_vimaan_intent") is not None and sec is not None and sec <= VIMAAN_SEC
            )
            who = (
                f"Vimaan ({ev.get('last_vimaan_cmd')}, {sec} s after)"
                if vimaan
                else ("the probe" if probe_on else "not Vimaan")
            )
            off = _went_off(changed)
            unexplained = off and not vimaan and not probe_on
            if unexplained and not drops_seen_by_plugin:
                ap_drops += 1
            items.append(
                _event(
                    ev,
                    "Autopilot off" if off else "Autopilot change",
                    "warn" if unexplained else "info",
                    changes=_autopilot_changes(changed),
                    who=who,
                    since=sec,
                    last_cmd=ev.get("last_vimaan_cmd"),
                    # The probe presses every mode in turn: details, not news.
                    level="system" if probe_on and not vimaan else "autopilot",
                )
            )
        elif kind == "autopilot_disconnect":
            silent = ev.get("said") is None
            if not silent:
                ap_drops += 1
            items.append(
                _event(
                    ev,
                    f"Autopilot disconnected: {ev.get('cause')}"
                    + (f" ({ev.get('detail')})" if ev.get("detail") else ""),
                    "info" if silent else "bad",
                    cause=ev.get("cause"),
                    said=ev.get("said"),
                    stick_peak=ev.get("stick_peak"),
                    nullzone=ev.get("nullzone"),
                    commands=ev.get("commands"),
                    flying={
                        k: ev.get(k)
                        for k in ("altitude_actual", "vs_actual", "ias_actual", "heading_actual")
                    },
                )
            )
        elif kind == "probe_start":
            probe_on = True
            items.append(
                _event(
                    ev,
                    f"Probe started ({ev.get('where')}, {ev.get('steps')} steps)",
                    level="system",
                )
            )
        elif kind == "probe_end":
            probe_on = False
            steps = ev.get("steps") or {}
            counts = Counter(steps.values())
            items.append(
                _event(
                    ev,
                    f"Probe ended ({ev.get('reason')}): "
                    + ", ".join(f"{n} {v}" for v, n in counts.most_common()),
                    level="system",
                )
            )
        elif kind == "probe_paused":
            items.append(
                _event(ev, f"Probe paused: the autopilot dropped during {ev.get('label')}", "warn")
            )
        elif kind == "probe_step":
            items.append(_event(ev, f"Probe {ev.get('role')}: {ev.get('verdict')}", level="system"))
        elif kind in ("session_end", "plugin_stop") and not ended:
            ended = True
            items.append(_event(ev, "Session ended"))
        elif kind == "speak" and cmd is None and ev.get("origin") != "probe":
            items.append(_event(ev, f"Vimaan said: {ev.get('text')}", level="system"))

        if kind in PROBLEM_KINDS:
            text = _problem_text(ev)
            problems.append({"t": ev.get("t"), "clock": _clock(ev), "kind": kind, "text": text})
            if not (isinstance(cmd, str) and cmd in commands):
                items.append(
                    _event(ev, text, "bad", level="system" if kind == "sim_stall" else None)
                )

    rows_cmds = list(commands.values())
    for c in rows_cmds:
        group, label = OUTCOMES.get(c["outcome"] or "", ("other", c["outcome"] or "No result"))
        c["group"], c["label"] = group, label
    by_group = Counter(c["group"] for c in rows_cmds)
    latencies = [c["latency_ms"] for c in rows_cmds if isinstance(c["latency_ms"], (int, float))]
    last_t = rows[-1].get("t") if rows else None
    first_ts = next((r.get("ts") for r in rows if r.get("ts")), None)
    summary = {
        "started": first_ts,
        "duration_sec": last_t,
        "aircraft": aircraft,
        "model": model,
        "stt": stt,
        "commands": len(rows_cmds),
        "groups": {g: by_group.get(g, 0) for g in GROUPS},
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "latency_p95_ms": round(_percentile(latencies, 95), 1) if latencies else None,
        "autopilot_drops": ap_drops,
        "problems": len(problems),
        "self_test": self_test,
        "joystick": joystick,
        "recordings": sum(1 for c in rows_cmds if c["audio"]),
        "status": "ended" if ended else ("live" if live else "crashed"),
    }
    return {"summary": summary, "items": items, "problems": problems}


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[k]


def _problem_text(ev: dict[str, Any]) -> str:
    kind = ev.get("kind")
    if kind == "sim_stall":
        return f"X-Plane paused or froze for {ev.get('gap_ms', 0) / 1000:.1f} s"
    if kind == "worker_exit":
        return f"Voice engine exited (code {ev.get('returncode')})"
    if kind == "stt_fallback":
        return f"Speech recognition fell back to {ev.get('using')}"
    detail = ev.get("error") or ev.get("message") or ev.get("dataref") or ""
    return f"{kind.replace('_', ' ')}{': ' + str(detail) if detail else ''}"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _is_live(path: Path) -> bool:
    try:
        return time.time() - path.stat().st_mtime < LIVE_SEC
    except OSError:
        return False


def load(session: str) -> dict[str, Any] | None:
    """One session, parsed (cached until its file changes)."""
    path = events_path(session)
    if path is None:
        return None
    stamp = _stamp(path)
    if stamp is None:
        return None
    hit = _cache.get(path)
    live = _is_live(path)
    if hit is not None and hit[0] == stamp:
        detail = hit[1]
        if detail["summary"]["status"] != "ended":
            detail["summary"]["status"] = "live" if live else "crashed"
        return detail
    detail = parse(_read(path), live=live)
    detail["session"] = session
    detail["summary"]["session"] = session
    _cache[path] = (stamp, detail)
    return detail


def stamp(session: str) -> tuple[int, int] | None:
    path = events_path(session)
    return None if path is None else _stamp(path)


def list_flights() -> list[dict[str, Any]]:
    """Every session, newest first, with its summary."""
    out = []
    for session in _sessions():
        detail = load(session)
        if detail is not None:
            out.append(detail["summary"])
    return out
