"""The automated-test bridge (Phase A): a test runner on the same Mac drives
Vimaan and reads the sim through files, so the tier checklists can run with
no one at the controls.

    ~/Vimaan_Logs/test/
        bridge.on           the switch: the bridge works only while it exists
        inbox/<id>.json     a request (the runner writes a temp file, then renames)
        outbox/<id>.json    its result

A request is ``{"id": "...", "op": "...", ...}``:

    state                                     Vimaan, the aircraft and the sim now
    say       {"text": "..."} | {"wav": path}  a command, as if spoken: text goes
                                               to the model only, a WAV through
                                               recognition first. The result
                                               comes when the command is done and
                                               its read-backs are checked.
    read      {"datarefs": ["name", "name[2]"]}
    write     {"dataref": name, "value": v}
    command   {"name": ..., "phase": "once" | "begin" | "end"}
    place     {"latitude", "longitude", "elevation_ft", "heading_true", "speed_kt"}
    airport   {"icao": ...}
    aircraft  {"path": ...}

Commands from the bridge count as the pilot's, not Vimaan's (so a scripted
autopilot disconnect is diagnosed like a real one).

Pure Python, no X-Plane: the plugin does the sim calls.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

FLAG = "bridge.on"
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
OPS = ("state", "say", "read", "write", "command", "place", "airport", "aircraft")
# How many requests to take per poll (a say waits for its command).
TAKE_MAX = 20
# A say result waits this long after the command for its read-backs.
CHECKS_WAIT_SEC = 6.0
# A say with no result after this long is answered with a timeout.
SAY_TIMEOUT_SEC = 40.0
# Event kinds a say result is built from.
_SAY_KINDS = {
    "stage",
    "speak",
    "sim_command",
    "sim_write",
    "verify",
    "command_done",
    "safety_hold",
    "safety_override",
    "slot_rejected",
    "implausible",
    "ambiguous",
    "bare_command",
    "not_applicable",
    "worker_error",
    "stage_ended",
}


def folder(log_dir: str | os.PathLike[str]) -> Path:
    return Path(log_dir) / "test"


def paths(log_dir: str | os.PathLike[str]) -> tuple[Path, Path, Path]:
    """(switch file, inbox, outbox)."""
    root = folder(log_dir)
    return root / FLAG, root / "inbox", root / "outbox"


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not an object")
    return data


def take_requests(inbox: Path, limit: int = TAKE_MAX) -> list[dict[str, Any]]:
    """Read and remove waiting requests, oldest first. A file that can't be
    read comes back as a request with an ``error``."""
    try:
        files = sorted(
            (p for p in inbox.glob("*.json") if p.is_file()),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
        )
    except OSError:
        return []
    out = []
    for p in files[:limit]:
        try:
            req = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(req, dict):
                raise ValueError("not an object")
        except (OSError, ValueError) as exc:
            req = {"error": f"unreadable request: {exc}"}
        req["id"] = req.get("id") if ID_RE.match(str(req.get("id", ""))) else p.stem
        try:
            p.unlink()
        except OSError:
            pass
        out.append(req)
    return out


def problem(req: dict[str, Any]) -> str | None:
    """What's wrong with a request, or None."""
    if req.get("error"):
        return str(req["error"])
    if not ID_RE.match(str(req.get("id", ""))):
        return "bad id"
    op = req.get("op")
    if op not in OPS:
        return f"unknown op {op!r}"
    if op == "say":
        wav, text = req.get("wav"), req.get("text")
        if bool(wav) == bool(text):
            return "say needs exactly one of text or wav"
        if wav and not (os.path.isabs(str(wav)) and str(wav).endswith(".wav")):
            return "wav must be an absolute path to a .wav file"
        if wav and not os.path.isfile(str(wav)):
            return f"no such file: {wav}"
    elif op == "read":
        names = req.get("datarefs")
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            return "read needs a list of dataref names"
    elif op == "write":
        if not isinstance(req.get("dataref"), str) or not isinstance(
            req.get("value"), (int, float, list)
        ):
            return "write needs a dataref name and a number (or a list)"
    elif op == "command":
        if not isinstance(req.get("name"), str) or req.get("phase", "once") not in (
            "once",
            "begin",
            "end",
        ):
            return "command needs a name, and phase once / begin / end"
    elif op == "place":
        for key in ("latitude", "longitude", "elevation_ft", "heading_true", "speed_kt"):
            if not isinstance(req.get(key), (int, float)):
                return f"place needs {key}"
    elif op == "airport":
        if not re.match(r"^[A-Z0-9]{3,7}$", str(req.get("icao", ""))):
            return "airport needs an ICAO code"
    elif op == "aircraft":
        if not str(req.get("path", "")).endswith(".acf"):
            return "aircraft needs the path of an .acf file"
    return None


def parse_dataref(name: str) -> tuple[str, int | None]:
    """ "sim/x/y[2]" -> ("sim/x/y", 2); "sim/x/y" -> ("sim/x/y", None)."""
    m = re.match(r"^(.+?)\[(\d+)\]$", name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


class SayResults:
    """Collects what happens to each bridged command, from the telemetry
    stream, and hands back finished results."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self._pending: dict[str, dict[str, Any]] = {}

    @property
    def busy(self) -> bool:
        return bool(self._pending)

    def start(self, rid: str, cmd: str, now: float) -> None:
        self._pending[cmd] = {"id": rid, "events": [], "started": now, "done_at": None}

    def on_event(self, ev: dict[str, Any]) -> None:
        p = self._pending.get(str(ev.get("cmd")))
        if p is None or ev.get("kind") not in _SAY_KINDS:
            return
        p["events"].append(ev)
        if ev.get("kind") == "command_done":
            p["done_at"] = self.clock()

    def finished(self, now: float, checking: Iterable[str]) -> list[dict[str, Any]]:
        """Results that are complete: the command is done and none of its
        read-backs are still due (or they've had long enough), or it timed out."""
        checking = set(checking)
        out = []
        for cmd, p in list(self._pending.items()):
            done = p["done_at"]
            if done is not None and (cmd not in checking or now - done > CHECKS_WAIT_SEC):
                out.append(summarize(p["id"], cmd, p["events"]))
            elif done is None and now - p["started"] > SAY_TIMEOUT_SEC:
                out.append(
                    {**summarize(p["id"], cmd, p["events"]), "ok": False, "error": "timeout"}
                )
            else:
                continue
            del self._pending[cmd]
        return out


def summarize(rid: str, cmd: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """One bridged command's result, from its events."""
    r: dict[str, Any] = {
        "id": rid,
        "ok": True,
        "cmd": cmd,
        "heard": None,
        "intent": None,
        "slots": {},
        "confidence": None,
        "outcome": None,
        "replies": [],
        "actions": [],
        "checks": [],
        "notes": [],
        "latency_ms": None,
    }
    for ev in events:
        kind = ev.get("kind")
        if kind == "stage" and ev.get("stage") == "stt":
            r["heard"] = ev.get("text")
        elif kind == "stage" and ev.get("stage") == "nlu":
            r["intent"], r["slots"] = ev.get("intent"), ev.get("slots") or {}
            r["confidence"] = ev.get("confidence")
        elif kind == "stage" and ev.get("stage") == "dispatch" and r["outcome"] is None:
            r["outcome"] = ev.get("outcome")
        elif kind == "speak" and ev.get("text"):
            r["replies"].append(ev["text"])
        elif kind == "sim_command":
            r["actions"].append({"command": ev.get("command"), "ok": ev.get("ok")})
        elif kind == "sim_write":
            r["actions"].append(
                {"dataref": ev.get("dataref"), "value": ev.get("value"), "ok": ev.get("ok")}
            )
        elif kind == "verify":
            r["checks"].append({k: ev.get(k) for k in ("label", "expected", "actual", "ok")})
        elif kind == "command_done":
            r["outcome"] = ev.get("outcome") or r["outcome"]
            r["latency_ms"] = ev.get("speech_to_action_ms")
            if r["heard"] is None:
                r["heard"] = ev.get("text")
        else:
            r["notes"].append(
                {k: v for k, v in ev.items() if k not in ("ts", "t", "thread", "cmd")}
            )
    return r
