"""What the in-sim overlay shows, as a pure state machine.

The overlay answers, at a glance, *where* a command went wrong: what the
recogniser heard (live, while you talk), what the model understood and how
sure it was, and what happened (done / confirm / refused / the sim didn't
respond), with the last few commands under it, fading as they age. It is
driven by the same telemetry events that go to ``events.jsonl``
(``Telemetry.subscribe``), plus live partial transcripts that are too frequent
to log. No X-Plane or ImGui imports here, so every state is unit-testable;
``overlay.py`` draws it.

The output is a :class:`Panel` (phase O.5, design D "avionics, top left"): a
header of tags (status, where the command came from), labelled rows (HEARD,
ACTION, CONF, ...), and the history. ``lines()`` flattens it to text lines for
the fallback renderer, which has no shapes. Fonts in X-Plane and ImGui are
ASCII-only, so status is shown as words ("DONE", "CAUTION"), not symbols.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from vimaan_nlu.commands import SLOT_DISPLAY, command

HOLD_SEC = 6.0  # how long the result stays up after a command finishes
PROBE_HOLD_SEC = 12.0  # how long a finished probe's summary stays up
CHECKLIST_HOLD_SEC = 8.0  # how long a finished checklist stays up
CHECKLIST_RESULT_SEC = 2.5  # a command's result, mid-checklist, before the list comes back
HISTORY_LINES = 3  # past commands shown under the panel
HISTORY_SEC = 90.0  # how long a past command stays in the history
HISTORY_FADE_SEC = 20.0  # ...fading out over the last of it
WORKING_MAX_SEC = 120.0  # a command still working never holds the screen longer
CONFIDENCE_FLOOR = 0.55  # the plugin's: below it, nothing is done
ANSWER_SEC = 12.0  # the plugin's windows for "override" / "which?" answers
MODES = ("auto", "pinned", "off")  # auto: shows on activity, then fades

# Where a command came from (``ptt_press`` source): anything else is a key.
SOURCES = {"wake": "WAKE", "test": "TEST"}

PHASE_NAMES = {
    "parked": "PARKED",
    "taxi": "TAXI",
    "takeoff_roll": "TAKEOFF",
    "initial_climb": "CLIMB OUT",
    "climb": "CLIMB",
    "cruise": "CRUISE",
    "descent": "DESCENT",
    "approach": "APPROACH",
    "landing": "LANDING",
    "go_around": "GO AROUND",
    "rollout": "ROLLOUT",
}

# Tones are mapped to colours by the renderer.
# outcome -> (status tag, history tag or None for none, tone)
OUTCOMES: dict[str, tuple[str, str | None, str]] = {
    "executed": ("DONE", "DONE", "ok"),
    "no_op": ("NO CHANGE", "SAME", "ok"),
    "pending": ("WORKING", "DONE", "accent"),
    "safety_hold": ("CAUTION", "HOLD", "warn"),
    "ambiguous": ("WHICH?", "ASK", "warn"),
    "answered": ("ANSWER", "ANS", "ok"),
    "low_confidence": ("REPEAT", "RPT", "warn"),
    "rejected": ("REFUSED", "REF", "warn"),
    "non_actionable": ("INFO", "INFO", "muted"),
    "no_handler": ("UNKNOWN", "UNK", "warn"),
    "not_understood": ("NOT HEARD", "MISS", "error"),
    "silent_mic": ("NO MIC", "MIC", "error"),
    "too_short": ("TOO SHORT", None, "muted"),
    "no_command": ("NO COMMAND", None, "muted"),
    "mic_error": ("MIC ERROR", "MIC", "error"),
    "stt_service_error": ("STT ERROR", "ERR", "error"),
    "timeout": ("TIMEOUT", "TIME", "error"),
    "cancelled": ("CANCELLED", "CNX", "warn"),
}
FAILED = ("ERROR", "ERR", "error")  # an outcome not in the table ("failed")
# Outcomes that did what was asked: their action is shown as done.
DONE_OUTCOMES = frozenset({"executed", "no_op", "pending", "answered"})
# Outcomes whose spoken reply says something the rows don't.
QUIET_REPLY = frozenset({"executed", "pending", "safety_hold", "ambiguous", "answered"})

REASONS: dict[str, str] = {
    "no_command_word": "no command word heard",
    "one_digit_heading": "heading needs three digits",
    "bare_number": "a number alone is not a command",
    "not_offered": "not one of the two values offered",
    "question": "questions aren't answered yet",
}

# What a command does, as the ACTION row (and the history) says it:
# "{slot}" is the slot's value, "{a|b}" the first of them given.
ACTIONS: dict[str, str] = {
    "set_autopilot_heading": "HDG {degrees}",
    "set_autopilot_altitude": "ALT {altitude} FT",
    "set_flight_level": "FL {flight_level}",
    "turn_by": "TURN {turn} {turn_degrees}",
    "set_compass_heading": "HDG {compass}",
    "run_checklist": "{checklist} CHECKLIST",
    "change_altitude_by": "ALT {direction} {altitude_change} FT",
    "change_speed_by": "SPD {direction} {speed_change} KT",
    "set_com_frequency": "COM{com_port} {frequency}",
    "toggle_landing_gear": "GEAR {state}",
    "set_flaps": "FLAPS {flap_setting|position}",
    "set_autopilot_speed": "SPD {speed|mach}",
    "set_vertical_speed": "V/S {direction} {vs} FPM",
    "set_autopilot_mode": "{mode} MODE {state}",
    "toggle_autothrottle": "A/THR {state}",
    "set_altimeter": "BARO {pressure|setting}",
    "toggle_autopilot": "AP{instance} {state}",
    "toggle_flight_director": "FD{instance} {state}",
    "toggle_parking_brake": "PARK BRAKE {state}",
    "toggle_engine": "ENG {instance} {state}",
    "set_com_standby": "COM{com_port} STBY {frequency}",
    "swap_com_frequency": "COM{com_port} FLIP",
    "set_nav_frequency": "NAV{nav_port} {nav_frequency}",
    "set_nav_standby": "NAV{nav_port} STBY {nav_frequency}",
    "swap_nav_frequency": "NAV{nav_port} FLIP",
    "set_course": "CRS{nav_port} {degrees}",
    "set_transponder_code": "SQUAWK {squawk}",
    "set_transponder_mode": "XPDR {xpdr_mode}",
    "toggle_lights": "{light} LIGHTS {switch}",
    "toggle_signs": "{sign} SIGN {switch}",
    "toggle_yaw_damper": "YAW DAMPER {state}",
    "set_speed_brake": "SPD BRK {speedbrake}",
    "set_autobrake": "AUTO BRK {autobrake|autobrake_level}",
    "toggle_anti_ice": "{ice} ANTI-ICE {switch}",
    "toggle_electrical": "{power} {instance} {switch}",
    "toggle_fuel_pump": "FUEL PUMP {instance} {switch}",
}
THOUSANDS = frozenset({"altitude", "altitude_change", "vs"})  # 6000 -> 6,000
_FIELD = re.compile(r"\{([a-z_|]+)\}")


@dataclass(frozen=True)
class Tag:
    """A boxed word in the header: filled (the status) or outlined."""

    text: str
    tone: str
    filled: bool = True


@dataclass(frozen=True)
class Row:
    """One labelled row. ``kind``: "text"; "conf" (a bar of ``value`` with the
    confidence floor marked); "bar" (a countdown or progress of ``value``);
    "spinner" (work under way, no end known); "item" (a checklist line,
    ``detail`` its reading); "offer" (the item being offered, ``text`` the
    offer); "note" (small, unlabelled)."""

    label: str
    text: str = ""
    tone: str = "text"
    kind: str = "text"
    value: float | None = None
    detail: str = ""
    detail_tone: str = "muted"


@dataclass(frozen=True)
class Past:
    """A past command in the history: ``age`` ("-0:12"), a short tag, what it
    did, and ``alpha`` (1, fading to 0 at the end of its time)."""

    age: str
    tag: str
    tone: str
    text: str
    alpha: float = 1.0


@dataclass(frozen=True)
class Panel:
    tags: tuple[Tag, ...] = ()
    title: str = ""  # after the tags (a hint, or a checklist's name)
    title_tone: str = "muted"
    meta: str = ""  # right of the header (latency, a countdown, the phase)
    meta_tone: str = "muted"
    alert: str = ""  # the border's tone when it needs attention ("warn")
    rows: tuple[Row, ...] = ()
    history: tuple[Past, ...] = ()
    pinned: bool = False
    floor: float = CONFIDENCE_FLOOR


@dataclass(frozen=True)
class Line:
    """A flat text line (the fallback renderer, and the probe's own view)."""

    badge: str  # short status label, drawn in the tone colour ("" for none)
    text: str
    tone: str  # live | heard | text | muted | accent | ok | warn | error
    detail: str = ""  # right-aligned meta (timings, a count)
    progress: float | None = None  # 0..1: a thin bar under the text (probe)


# The aircraft probe's result for a step -> (badge, tone).
PROBE_BADGE: dict[str, tuple[str, str]] = {
    "works": ("WORKS", "ok"),
    "no_change": ("NO CHANGE", "warn"),
    "missing": ("MISSING", "muted"),
    "inconclusive": ("UNCLEAR", "muted"),
}


def describe_intent(intent: str) -> str:
    cmd = command(intent)
    return cmd.display if cmd else intent.replace("_", " ").capitalize()


def describe_slots(slots: dict[str, Any]) -> str:
    parts = []
    for name, value in slots.items():
        if str(value).isalpha():  # on / up / both
            parts.append(str(value).upper())
        else:
            parts.append(f"{SLOT_DISPLAY.get(name, name.replace('_', ' '))} {value}")
    return "  ".join(parts)


def _slot_value(name: str, value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[a-z]( [a-z])+", text):  # "r t o", as said
        text = text.replace(" ", "")
    if name in THOUSANDS and text.isdigit():
        text = f"{int(text):,}"
    return text.upper()


def action_text(intent: str, slots: dict[str, Any]) -> str:
    """What the command does, short and in capitals ("HDG 270",
    "LANDING LIGHTS ON"). Commands without a format say their name and
    values."""
    fmt = ACTIONS.get(intent)
    if fmt is None:
        values = " ".join(_slot_value(k, v) for k, v in slots.items())
        return f"{describe_intent(intent).upper()} {values}".strip()

    def field(m: re.Match[str]) -> str:
        for name in m.group(1).split("|"):
            if slots.get(name) not in (None, ""):
                return _slot_value(name, slots[name])
        return ""

    return " ".join(_FIELD.sub(field, fmt).split())


def age_text(sec: float) -> str:
    sec = max(0, int(sec))
    return f"-{sec // 60}:{sec % 60:02d}"


class OverlayState:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        mode: str = "auto",
        *,
        floor: float = CONFIDENCE_FLOOR,
        answer_sec: float = ANSWER_SEC,
        working: Callable[[], Iterable[str]] | None = None,
    ) -> None:
        self.clock = clock
        self.mode = mode if mode in MODES else "auto"
        self.floor = floor
        self.answer_sec = answer_sec
        # The commands with work still due (a switch stepping, an engine
        # starting), from the plugin; None: known only from their end events.
        self.working = working
        self._reset(None)
        self.phase = "idle"  # idle | listening | processing | done | notice
        self.done_at: float | None = None
        self.notice_tone = "text"  # a callout's notice is amber
        # The aircraft probe has its own view, fed by its own events, so its
        # results never mix with the last voice command's lines.
        self.probe: dict[str, Any] | None = None
        self.checklist: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []  # oldest first
        self.wake: bool | None = None  # the wake word on (None: not known yet)
        self.callouts = False
        self.suggestions = False
        self.flight_phase: str | None = None

    def _reset(self, cmd: str | None) -> None:
        self.cmd = cmd
        self.source = "KEY"
        self.partial = ""
        self.heard: str | None = None
        self.backend: str | None = None
        self.stt_ms: float | None = None
        self.intent: str | None = None
        self.slots: dict[str, Any] = {}
        self.confidence: float | None = None
        self.reply: str | None = None
        self.outcome: str | None = None
        self.reason: str | None = None
        self.verified: bool | None = None
        self.latency_ms: float | None = None
        self.hold: dict[str, Any] | None = None  # {"t", "message", "expired"}
        self.question: dict[str, Any] | None = None  # {"t", "expired"}
        self.work: dict[str, Any] | None = None  # {"t", "ended", "failed"}
        self.dropped: list[str] = []
        self.entry: dict[str, Any] | None = None  # this command's history line
        self.hold_sec = HOLD_SEC  # how long its result stays up

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def on_partial(self, cmd: str | None, text: str) -> None:
        if self.phase in ("listening", "processing") and cmd == self.cmd:
            self.partial = text

    def on_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("kind")
        cmd = ev.get("cmd")
        if self._on_setting(kind, ev):
            return
        if str(kind).startswith("probe_"):
            self._on_probe(ev)
            return
        if str(kind).startswith("checklist_"):
            self._on_checklist(ev)
            return
        if kind == "speak" and ev.get("origin") == "probe":
            return  # the probe view shows it; it isn't a reply to a command
        if kind == "ptt_press":
            self._reset(cmd)
            self.source = SOURCES.get(str(ev.get("source")), "KEY")
            self.phase = "listening"
            return
        if kind == "callout":
            if self.phase not in ("listening", "processing"):
                self._reset(None)
                self.reply = ev.get("said")
                self.phase = "notice"
                self.notice_tone = "warn"
                self.done_at = self.clock()
            return
        if kind == "speak":
            self._on_speak(ev)
            return
        if cmd is not None and cmd != self.cmd:
            return
        self._on_command_event(kind, ev)

    def _on_setting(self, kind: Any, ev: dict[str, Any]) -> bool:
        if kind == "wake_word_setting":
            self.wake = bool(ev.get("on"))
        elif kind == "callouts_setting":
            self.callouts = bool(ev.get("enabled"))
        elif kind == "checklist_suggestions_setting":
            self.suggestions = bool(ev.get("enabled"))
        elif kind == "flight_phase":
            self.flight_phase = ev.get("phase")
        else:
            return False
        return True

    def _on_speak(self, ev: dict[str, Any]) -> None:
        # A checklist's own words: its view shows them.
        run = self.checklist
        if (
            run is not None
            and ev.get("cmd") == run["cmd"]
            and self.phase not in ("listening", "processing")
        ):
            return
        # Spoken replies carry a stale or missing cmd on some paths ("I could
        # not understand you", engine start results), so take them whenever
        # the overlay is in use, and as a notice otherwise.
        text = ev.get("text")
        if self.phase == "notice" and self.notice_tone == "warn" and text == self.reply:
            return  # the callout said
        self.reply = text
        if self.phase == "idle":
            self._reset(None)
            self.reply = text
            self.phase = "notice"
            self.notice_tone = "text"
        if self.phase in ("done", "notice") and self.work is None:
            self.done_at = self.clock()

    def _on_command_event(self, kind: Any, ev: dict[str, Any]) -> None:
        now = self.clock()
        if kind == "ptt_release" and self.phase == "listening":
            self.phase = "processing"
        elif kind == "stage" and ev.get("stage") == "stt":
            self.heard = ev.get("text")
            self.backend = ev.get("backend")
            self.stt_ms = ev.get("ms")
        elif kind == "stage" and ev.get("stage") == "nlu":
            self.intent = ev.get("intent")
            self.slots = dict(ev.get("slots") or {})
            self.confidence = ev.get("confidence")
        elif kind in ("answered", "safety_override", "question_answered", "checklist_confirmed"):
            # What it was taken as: a bare number answering "to or two?", the
            # held command overridden, the question answered (a bare "heading"
            # the model read as heading mode is answered as the heading).
            self.intent = ev.get("intent")
            self.slots = dict(ev.get("slots") or {})
            self.confidence = None
        elif kind == "to_reading":
            # Impossible as heard: show the "to" reading Vimaan asks about.
            self.intent = ev.get("intent")
            self.slots = dict(ev.get("slots") or {})
            self.confidence = None
            self.reason = 'a "two" taken as "to"'
            if not ev.get("repeat"):
                self.question = {"t": now, "expired": False}
        elif kind == "ambiguous":
            self.question = {"t": now, "expired": False}
        elif kind == "implausible":
            self.reason = REASONS.get(ev.get("reason", ""), ev.get("reason"))
        elif kind == "slot_rejected":
            self.reason = f"{ev.get('slot')} {ev.get('reason')}"
        elif kind == "safety_hold":
            rule = str(ev.get("rule", "")).replace("_", " ")
            self.hold = {"t": now, "message": ev.get("message") or rule, "expired": False}
        elif kind == "cancelled":
            self.dropped = [str(d) for d in ev.get("dropped") or ()]
        elif kind == "command_done":
            self._on_done(ev, now)
        elif kind == "verify":
            self.verified = bool(ev.get("ok")) and self.verified is not False
            if self.verified is False:
                self._update_entry("FAIL", "error")
            if self.work is None:
                self.done_at = now
        elif kind in (
            "heading_step_end",
            "baro_step_end",
            "knob_step_end",
            "crank_end",
            "apu_start_end",
        ):
            ok = ev.get("ok", ev.get("result") == "running")
            if self.work is not None:
                self.work["ended"] = True
                self.work["failed"] = self.work["failed"] or not ok
            if not ok:
                self._update_entry("FAIL", "error")
            self.done_at = now

    def _on_done(self, ev: dict[str, Any], now: float) -> None:
        self.outcome = ev.get("outcome")
        self.latency_ms = ev.get("speech_to_action_ms")
        if self.heard is None and ev.get("text"):
            self.heard = ev.get("text")
        self.phase = "done"
        self.done_at = now
        # Mid-checklist, the list comes back soon (it goes on reading).
        self.hold_sec = CHECKLIST_RESULT_SEC if self._checklist_running() else HOLD_SEC
        if self.outcome == "pending":
            self.work = {"t": now, "ended": False, "failed": False}
        if self.outcome not in ("ambiguous",):
            self.question = None
        if self.outcome != "safety_hold":
            self.hold = None
        self._add_history(now)

    def _add_history(self, now: float) -> None:
        _status, tag, tone = OUTCOMES.get(self.outcome or "", FAILED)
        if tag is None:
            return
        if self.intent == "run_checklist" and self.outcome == "executed":
            return  # the checklist adds its own line when it ends
        text = self._history_text()
        self.entry = {"t": now, "cmd": self.cmd, "tag": tag, "tone": tone, "text": text}
        self.history.append(self.entry)
        del self.history[:-10]

    def _history_text(self) -> str:
        if self.outcome == "answered" and self.reply:
            return self.reply.upper()
        if self.outcome == "cancelled":
            what = [action_text(d, {}) if command(d) else d.upper() for d in self.dropped]
            return ", ".join(what) or "NOTHING"
        if (
            self.intent
            and self.intent != "None"
            and self.outcome
            in DONE_OUTCOMES
            | {
                "safety_hold",
                "ambiguous",
                "rejected",
                "low_confidence",
            }
        ):
            return action_text(self.intent, self.slots)
        if self.heard:
            return f'"{self.heard.upper()}"'
        return (self.reply or "").upper()

    def _update_entry(self, tag: str, tone: str, suffix: str = "") -> None:
        if self.entry is not None:
            self.entry.update(tag=tag, tone=tone)
            if suffix and not self.entry["text"].endswith(suffix):
                self.entry["text"] += suffix

    def _on_checklist(self, ev: dict[str, Any]) -> None:
        kind, run, now = ev.get("kind"), self.checklist, self.clock()
        if kind == "checklist_start":
            self.checklist = {
                "name": str(ev.get("checklist") or ""),
                "cmd": ev.get("cmd"),
                "items": [str(i) for i in ev.get("items") or ()],
                "read": {},  # item -> (reading, result)
                "offer": None,  # (item, ask)
                "ended_at": None,
                "reason": None,
                "open": [],
            }
            return
        if kind == "checklist_confirmed":
            if ev.get("cmd") == self.cmd:
                self._on_command_event(kind, ev)
            return
        if run is None or run["ended_at"] is not None:
            return
        item = str(ev.get("item") or "")
        if kind == "checklist_item":
            result = str(ev.get("result") or "")
            run["read"][item] = (ev.get("state"), result)
            if result == "offered":
                run["offer"] = (item, ev.get("ask") or "")
        elif kind == "checklist_offer":
            run["read"][item] = (run["read"].get(item, (None, ""))[0], str(ev.get("result")))
            run["offer"] = None
        elif kind == "checklist_end":
            run.update(ended_at=now, reason=ev.get("reason"), open=list(ev.get("open") or ()))
            run["offer"] = None
            if ev.get("reason") != "replaced":
                tag, tone = "CHKL", "accent"
                summary = self._checklist_summary(run)
                self.history.append(
                    {
                        "t": now,
                        "cmd": run["cmd"],
                        "tag": tag,
                        "tone": tone,
                        "text": f"{run['name'].upper()}, {summary[0]}",
                    }
                )
                del self.history[:-10]

    @staticmethod
    def _checklist_summary(run: dict[str, Any]) -> tuple[str, str]:
        if run["reason"] not in ("done", None):
            return "STOPPED", "muted"
        left = list(dict.fromkeys(run["open"]))
        if not left:
            return "COMPLETE", "ok"
        return f"{len(left)} OPEN", "warn"

    def _on_probe(self, ev: dict[str, Any]) -> None:
        kind, p = ev.get("kind"), self.probe
        if kind == "probe_start":
            self.probe = {
                "name": ev.get("name") or "",
                "where": ev.get("where") or "",
                "total": int(ev.get("steps") or 0),
                "done": int(ev.get("start") or 0),  # a resumed run starts part way
                "step": None,
                "status": "ready",
                "next": ev.get("next"),
                "counts": {},
                "message": None,
                "note": None,
                "ended_at": None,
            }
        elif p is None:
            return
        elif kind == "probe_act":
            p.update(step=ev.get("label"), status="testing", note=None, message=None)
        elif kind == "probe_step":
            verdict = ev.get("verdict") or ""
            p["counts"][verdict] = p["counts"].get(verdict, 0) + 1
            p.update(
                step=ev.get("label"),
                status=verdict,
                done=int(ev.get("index", p["done"])) + 1,
                next=ev.get("next"),
                note=ev.get("note"),
            )
        elif kind == "probe_paused":
            p.update(step=ev.get("label"), status="paused", next=ev.get("label"))
        elif kind == "probe_blocked":
            p.update(status="blocked", message=ev.get("message"), next=ev.get("label"))
        elif kind == "probe_move":
            p.update(
                status="ready",
                step=None,
                note=None,
                done=int(ev.get("index") or 0),
                next=ev.get("label"),
            )
        elif kind == "probe_end":
            ok = ev.get("reason") == "complete"
            p.update(status="complete" if ok else "stopped", message=ev.get("message"))
            if ev.get("counts"):
                p["counts"] = dict(ev["counts"])  # every saved run, not just this one
            p["ended_at"] = self.clock()
        # Probe news replaces a finished command on screen (not one in progress).
        if self.phase in ("done", "notice") and not self._waiting():
            self.phase = "idle"

    def notice(self, text: str) -> None:
        """Show a one-line message (e.g. "Overlay pinned")."""
        if self.phase in ("listening", "processing") or self._waiting():
            return
        self._reset(None)
        self.reply = text
        self.phase = "notice"
        self.notice_tone = "text"
        self.done_at = self.clock()

    def cycle_mode(self) -> str:
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        return self.mode

    # ------------------------------------------------------------------
    # Time: holds and questions run out, work finishes
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        now = self.clock()
        for wait in (self.hold, self.question):
            if wait is not None and not wait["expired"] and now >= wait["t"] + self.answer_sec:
                wait["expired"] = True
                self.done_at = wait["t"] + self.answer_sec
                if wait is self.hold:
                    self._update_entry("HOLD", "muted", ", EXPIRED")
        work = self.work
        if work is not None and not work.get("finished"):
            if not self._still_working(now):
                work["finished"] = True
                self.done_at = max(self.done_at or now, now)

    def _still_working(self, now: float) -> bool:
        work = self.work
        if work is None or work.get("finished") or now - work["t"] >= WORKING_MAX_SEC:
            return False
        if self.working is not None:
            try:
                return self.cmd in set(self.working())
            except Exception:
                return False
        return not work["ended"]

    def _waiting(self) -> bool:
        """A hold or a question open, or a command still working: never
        faded, never replaced by news."""
        if self.phase != "done":
            return False
        for wait in (self.hold, self.question):
            if wait is not None and not wait["expired"]:
                return True
        return self.work is not None and not self.work.get("finished")

    def _probe_showing(self) -> bool:
        p = self.probe
        if p is None:
            return False
        return p["ended_at"] is None or self.clock() - p["ended_at"] < PROBE_HOLD_SEC

    def _checklist_showing(self) -> bool:
        run = self.checklist
        if run is None:
            return False
        return run["ended_at"] is None or self.clock() - run["ended_at"] < CHECKLIST_HOLD_SEC

    def _command_showing(self) -> bool:
        if self.phase in ("listening", "processing") or self._waiting():
            return True
        hold = self.hold_sec if self.phase == "done" else HOLD_SEC
        return (
            self.phase in ("done", "notice")
            and self.done_at is not None
            and self.clock() - self.done_at < hold
        )

    def _checklist_running(self) -> bool:
        return self.checklist is not None and self.checklist["ended_at"] is None

    def _history_live(self) -> list[dict[str, Any]]:
        now = self.clock()
        return [h for h in self.history if now - h["t"] < HISTORY_SEC]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    @property
    def visible(self) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "pinned":
            return True
        self._tick()
        return (
            self._command_showing()
            or self._checklist_showing()
            or self._probe_showing()
            or bool(self._history_live())
        )

    def panel(self) -> Panel:
        """What to draw now. A voice command (live, waiting for an answer, or
        just finished) takes the screen; then a notice; then a running
        checklist; then the probe; else the idle view with the history."""
        self._tick()
        pinned = self.mode == "pinned"
        if self._command_showing():
            if self.phase == "notice":
                return self._notice_panel(pinned)
            return self._command_panel(pinned)
        if self._checklist_showing():
            return self._checklist_panel(pinned)
        if self._probe_showing():
            return self._probe_panel(pinned)
        return self._idle_panel(pinned)

    def _past(self, skip_cmd: str | None = None) -> tuple[Past, ...]:
        now = self.clock()
        out = []
        for h in reversed(self._history_live()):
            if skip_cmd is not None and h["cmd"] == skip_cmd and h is self.entry:
                continue  # the command on screen
            age = now - h["t"]
            alpha = min(1.0, max(0.0, (HISTORY_SEC - age) / HISTORY_FADE_SEC))
            out.append(Past(age_text(age), h["tag"], h["tone"], h["text"], alpha))
            if len(out) == HISTORY_LINES:
                break
        return tuple(out)

    def _idle_panel(self, pinned: bool) -> Panel:
        tags: list[Tag] = []
        if self.wake:
            tags.append(Tag("WAKE", "ok", filled=False))
        if self.callouts:
            tags.append(Tag("CALLOUTS", "muted", filled=False))
        if self.suggestions:
            tags.append(Tag("SUGGEST", "muted", filled=False))
        if not tags:
            tags.append(Tag("VIMAAN", "muted", filled=False))
        hint = "HEY COPILOT" if self.wake else "HOLD THE TALK KEY"
        return Panel(
            tags=tuple(tags),
            title=hint,
            meta=PHASE_NAMES.get(self.flight_phase or "", ""),
            history=self._past(),
            pinned=pinned,
            floor=self.floor,
        )

    def _notice_panel(self, pinned: bool) -> Panel:
        callout = self.notice_tone == "warn"
        tag = Tag("CALLOUT", "warn") if callout else Tag("VIMAAN", "muted", filled=False)
        return Panel(
            tags=(tag,),
            rows=(Row("", (self.reply or "").upper(), "warn" if callout else "text"),),
            history=() if callout else self._past(),
            pinned=pinned,
            floor=self.floor,
        )

    def _status(self) -> tuple[Tag, str]:
        """The status tag and the border's tone."""
        now = self.clock()
        if self.phase == "listening":
            return Tag("LISTEN", "live"), ""
        if self.phase == "processing":
            return Tag("WAIT", "accent"), ""
        hold, question, work = self.hold, self.question, self.work
        if hold is not None:
            if hold["expired"]:
                return Tag("EXPIRED", "muted"), ""
            return Tag("CAUTION", "warn"), "warn"
        if question is not None:
            if question["expired"]:
                return Tag("NO ANSWER", "muted"), ""
            return Tag("WHICH?", "warn"), "warn"
        if self.verified is False:
            return Tag("NO RESPONSE", "error"), "error"
        if work is not None:
            if work["failed"]:
                return Tag("FAILED", "error"), "error"
            if self._still_working(now):
                return Tag("WORKING", "accent"), ""
            return Tag("DONE", "ok"), ""
        status, _tag, tone = OUTCOMES.get(self.outcome or "", FAILED)
        return Tag(status, tone), "warn" if self.outcome == "low_confidence" else ""

    def _command_panel(self, pinned: bool) -> Panel:
        now = self.clock()
        status, alert = self._status()
        tags = (status, Tag(self.source, "ok" if self.source == "WAKE" else "muted", False))
        rows: list[Row] = []
        meta, meta_tone = "", "muted"
        listening = self.phase == "listening"
        # HEARD: live while talking, then the transcript.
        if listening:
            rows.append(Row("HEARD", f"{self.partial}_", "heard"))
        elif self.heard is not None:
            rows.append(Row("HEARD", self.heard, "heard"))
        elif self.phase == "processing":
            rows.append(Row("HEARD", self.partial or "...", "heard"))
        else:
            rows.append(Row("HEARD", "--", "muted"))
        # ACTION / ANSWER, then CONF.
        if self.outcome == "answered":
            rows.append(Row("ANSWER", (self.reply or "").upper(), "text"))
        elif listening or (self.phase == "processing" and self.intent is None):
            rows.append(Row("ACTION", "----", "muted"))
        elif self.intent is not None and self.intent != "None":
            done = self.outcome in DONE_OUTCOMES or self.phase != "done"
            text = action_text(self.intent, self.slots)
            if self.outcome == "low_confidence":
                text += "?"
            failed = self.phase == "done" and not done and self.hold is None
            rows.append(
                Row(
                    "ACTION",
                    text,
                    "text" if done or self.hold is not None else "muted",
                    detail="NOT DONE" if failed and self.question is None else "",
                    detail_tone="warn",
                )
            )
        elif self.phase == "done":
            rows.append(Row("ACTION", "NONE, NOTHING CHANGED", "muted"))
        if listening:
            rows.append(Row("CONF", kind="conf", tone="muted", detail="--"))
        elif self.confidence is not None:
            weak = self.confidence < self.floor
            tone = "warn" if weak else "ok"
            pct = f"{self.confidence:.0%}"
            rows.append(
                Row("CONF", "", tone, "conf", self.confidence, detail=pct, detail_tone=tone)
            )
        # What is waited for, or what happened.
        hold, question = self.hold, self.question
        if hold is not None:
            left = max(0.0, hold["t"] + self.answer_sec - now)
            if hold["expired"]:
                rows.append(Row("HOLD", str(hold["message"]).upper(), "muted"))
                rows.append(Row("", "NOT DONE: CONFIRMATION EXPIRED", "muted", kind="note"))
            else:
                rows.append(Row("HOLD", str(hold["message"]).upper(), "warn"))
                rows.append(Row("SAY", "OVERRIDE OR REPEAT", "text"))
                rows.append(Row("", kind="bar", tone="warn", value=left / self.answer_sec))
                rows.append(Row("", "AT 0: NOT DONE, CONFIRMATION EXPIRES", "muted", kind="note"))
                meta, meta_tone = f"{math.ceil(left)} S", "warn"
        elif question is not None:
            left = max(0.0, question["t"] + self.answer_sec - now)
            rows.append(
                Row(
                    "ASK",
                    (self.reply or "").upper(),
                    "warn" if not question["expired"] else "muted",
                )
            )
            if question["expired"]:
                rows.append(Row("", "NO ANSWER: NOTHING CHANGED", "muted", kind="note"))
            else:
                rows.append(Row("", kind="bar", tone="warn", value=left / self.answer_sec))
                meta, meta_tone = f"{math.ceil(left)} S", "warn"
        elif self.phase == "done":
            # The reply, unless it only says what ACTION says.
            said = (self.reply or "").upper()
            if (
                self.intent
                and self.intent != "None"
                and said == action_text(self.intent, self.slots)
            ):
                said = ""
            if self.work is not None and self._still_working(now):
                if said:
                    rows.append(Row("REPLY", said, "text"))
                rows.append(Row("", kind="spinner", tone="accent"))
            elif said and (self.outcome not in QUIET_REPLY or self.verified is False or self.work):
                tone = "error" if status.tone == "error" else "text"
                rows.append(Row("REPLY", said, tone))
            if self.latency_ms is not None and not self._still_working(now):
                meta = f"{self.latency_ms:.0f} MS"
        if self.phase == "done" and self.reason:
            rows.append(Row("NOTE", self.reason.upper(), "muted"))
        attention = bool(alert) or (self.work is not None and self._still_working(now))
        return Panel(
            tags=tags,
            meta=meta,
            meta_tone=meta_tone,
            alert=alert,
            rows=tuple(rows),
            history=() if attention else self._past(skip_cmd=self.cmd),
            pinned=pinned,
            floor=self.floor,
        )

    def _checklist_panel(self, pinned: bool) -> Panel:
        run = self.checklist
        assert run is not None
        rows: list[Row] = []
        offer = run["offer"]
        for item in run["items"]:
            name = item.upper()
            if offer is not None and offer[0] == item:
                reading = run["read"].get(item, (None, ""))[0]
                ask = str(offer[1]).upper()
                rows.append(
                    Row(
                        name,
                        f"{ask}? SAY CONFIRM / CANCEL" if ask else "SAY CONFIRM / CANCEL",
                        "accent",
                        kind="offer",
                        detail=str(reading or "--").upper(),
                        detail_tone="text",
                    )
                )
                continue
            if item not in run["read"]:
                rows.append(Row(name, kind="item", tone="muted", detail="--"))
                continue
            reading, result = run["read"][item]
            value = "CAN'T READ" if reading is None else str(reading).upper()
            if result == "set":  # offered and set
                tone, value = "ok", "SET"
            elif result == "checked":
                tone = "ok"
            else:  # wrong: left open, or the offer declined
                tone = "warn"
                if reading is not None:
                    value += ", NOT SET"
            rows.append(Row(name, kind="item", tone=tone, detail=value, detail_tone=tone))
        read = len(run["read"])
        meta = f"{read}/{len(run['items'])}"
        if run["ended_at"] is not None:
            summary, tone = self._checklist_summary(run)
            rows.append(Row("", summary, tone, kind="note"))
        return Panel(
            tags=(Tag("CHKL", "accent"),),
            title=run["name"].upper(),
            title_tone="text",
            meta=meta,
            rows=tuple(rows),
            pinned=pinned,
            floor=self.floor,
        )

    def _probe_panel(self, pinned: bool) -> Panel:
        head, *rest = self._probe_lines()
        rows: list[Row] = []
        if head.progress is not None:
            rows.append(Row("", kind="bar", tone="accent", value=head.progress))
        rows += [Row(ln.badge, ln.text, ln.tone) for ln in rest]
        return Panel(
            tags=(Tag("PROBE", "accent"),),
            title=head.text.upper(),
            title_tone="text",
            meta=head.detail,
            rows=tuple(rows),
            pinned=pinned,
            floor=self.floor,
        )

    def _probe_lines(self) -> list[Line]:
        p = self.probe
        assert p is not None
        total, done = p["total"], p["done"]
        title = "  ".join(x for x in (p["name"], f"{p['where']} probe") if x)
        out = [
            Line(
                "PROBE",
                title,
                "accent",
                f"{done} / {total}" if total else "",
                progress=done / total if total else None,
            )
        ]
        status, step = p["status"], p["step"]
        if status == "complete":
            counts = p["counts"]
            summary = ", ".join(
                f"{counts[k]} {label}"
                for k, label in (
                    ("works", "work"),
                    ("no_change", "no change"),
                    ("missing", "missing"),
                    ("inconclusive", "unclear"),
                )
                if counts.get(k)
            )
            out.append(Line("DONE", f"Results saved. {summary}.", "ok"))
            return out
        if status == "stopped":
            out.append(Line("STOPPED", p["message"] or "Probe stopped", "error"))
            return out
        if status == "testing":
            out.append(Line("TESTING", step or "", "live"))
        elif status == "paused":
            out.append(Line("PAUSED", f"Autopilot disconnected during {step}", "warn"))
            out.append(Line("NEXT", "Re-engage the autopilot, then press the probe key", "muted"))
            return out
        elif status == "blocked":
            out.append(Line("WAIT", p["message"] or "", "warn"))
        elif status in PROBE_BADGE:
            badge, tone = PROBE_BADGE[status]
            out.append(Line(badge, step or "", tone))
            if p["note"]:
                out.append(Line("NOTE", p["note"][:1].upper() + p["note"][1:], "muted"))
        if p["next"]:
            hint = "press the probe key" if status != "testing" else "after this step"
            out.append(Line("NEXT", f"{p['next']}  ({hint})", "muted"))
        return out

    def lines(self) -> list[Line]:
        """The panel as flat text lines, for the fallback renderer (no shapes:
        bars become their numbers or nothing)."""
        if not self._command_showing() and not self._checklist_showing():
            if self._probe_showing():
                return self._probe_lines()
        return flatten(self.panel())


def flatten(panel: Panel) -> list[Line]:
    """One line for the header (an unlabelled first row joins it: a notice),
    one per row and per past command."""
    head = panel.tags[0] if panel.tags else Tag("VIMAAN", "muted")
    rows = list(panel.rows)
    title = panel.title
    if not title and rows and not rows[0].label and rows[0].kind == "text":
        title = rows.pop(0).text
    out = [Line(head.text, title, head.tone, panel.meta)]
    for row in rows:
        if row.kind in ("bar", "spinner"):
            continue
        if row.kind == "item":
            text = row.detail
        elif row.kind == "offer":
            text = f"{row.detail}  {row.text}"
        elif row.kind == "conf":
            text = row.detail
        else:
            text = row.text
        detail = row.detail if row.kind == "text" else ""
        out.append(Line(row.label, text, row.tone, detail))
    for past in panel.history:
        out.append(Line(past.tag, f"{past.age}  {past.text}", past.tone))
    return out
