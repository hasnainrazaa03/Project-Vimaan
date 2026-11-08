"""What the in-sim overlay shows, as a pure state machine.

The overlay answers, at a glance, *where* a command went wrong: what the
recogniser heard (live, while you talk), what the model understood (intent,
values, confidence), and what happened (done / confirm / rejected / the sim
didn't respond). It is driven by the same telemetry events that go to
``events.jsonl`` (``Telemetry.subscribe``), plus live partial transcripts that
are too frequent to log. No X-Plane or ImGui imports here, so every state is
unit-testable; ``overlay.py`` draws it.

Fonts in X-Plane and ImGui's default font are ASCII-only, so status is shown
as text badges ("DONE", "CONFIRM?") rather than symbols.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vimaan_nlu.commands import SLOT_DISPLAY, command

HOLD_SEC = 6.0  # how long the result stays up after a command finishes
PROBE_HOLD_SEC = 12.0  # how long a finished probe's summary stays up
MODES = ("auto", "pinned", "off")  # auto: shows on activity, then fades

# Tones are mapped to colours by the renderer.
OUTCOME_BADGE: dict[str, tuple[str, str]] = {
    "executed": ("DONE", "ok"),
    "no_op": ("NO CHANGE", "ok"),
    "pending": ("WORKING", "accent"),
    "safety_hold": ("CONFIRM?", "warn"),
    "ambiguous": ("WHICH?", "warn"),
    "low_confidence": ("REPEAT", "warn"),
    "rejected": ("REJECTED", "warn"),
    "non_actionable": ("INFO", "muted"),
    "no_handler": ("UNKNOWN", "warn"),
    "not_understood": ("NOT HEARD", "error"),
    "silent_mic": ("NO MIC", "error"),
    "too_short": ("TOO SHORT", "muted"),
    "mic_error": ("MIC ERROR", "error"),
    "stt_service_error": ("STT ERROR", "error"),
    "timeout": ("TIMEOUT", "error"),
    "cancelled": ("CANCELLED", "muted"),
}

REASONS: dict[str, str] = {
    "no_command_word": "no command word heard",
    "one_digit_heading": "heading needs three digits",
    "bare_number": "a number alone is not a command",
    "not_offered": "not one of the two values offered",
}


@dataclass(frozen=True)
class Line:
    badge: str  # short status label, drawn in the tone colour ("" for none)
    text: str
    tone: str  # live | text | muted | accent | ok | warn | error
    detail: str = ""  # right-aligned meta (backend, timings)
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


class OverlayState:
    def __init__(self, clock: Callable[[], float] = time.monotonic, mode: str = "auto") -> None:
        self.clock = clock
        self.mode = mode if mode in MODES else "auto"
        self._reset(None)
        self.phase = "idle"  # idle | listening | processing | done | notice
        self.done_at: float | None = None
        # The aircraft probe has its own view, fed by its own events, so its
        # results never mix with the last voice command's lines.
        self.probe: dict[str, Any] | None = None

    def _reset(self, cmd: str | None) -> None:
        self.cmd = cmd
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

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def on_partial(self, cmd: str | None, text: str) -> None:
        if self.phase in ("listening", "processing") and cmd == self.cmd:
            self.partial = text

    def on_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("kind")
        cmd = ev.get("cmd")
        if str(kind).startswith("probe_"):
            self._on_probe(ev)
            return
        if kind == "speak" and ev.get("origin") == "probe":
            return  # the probe view shows it; it isn't a reply to a command
        if kind == "ptt_press":
            self._reset(cmd)
            self.phase = "listening"
            return
        if kind == "speak":
            # Spoken replies carry a stale or missing cmd on some paths
            # ("I could not understand you", engine start results), so take
            # them whenever the overlay is in use, and as a notice otherwise.
            self.reply = ev.get("text")
            if self.phase == "idle":
                self._reset(None)
                self.reply = ev.get("text")
                self.phase = "notice"
            if self.phase in ("done", "notice"):
                self.done_at = self.clock()
            return
        if kind == "crank_end":
            self.done_at = self.clock()
            return
        if cmd is not None and cmd != self.cmd:
            return
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
        elif kind in ("answered", "safety_override"):
            # A bare-number reply to "to or two?": show what it was taken as.
            self.intent = ev.get("intent")
            self.slots = dict(ev.get("slots") or {})
            self.confidence = None
        elif kind == "implausible":
            self.reason = REASONS.get(ev.get("reason", ""), ev.get("reason"))
        elif kind == "slot_rejected":
            self.reason = f"{ev.get('slot')} {ev.get('reason')}"
        elif kind == "safety_hold":
            self.reason = str(ev.get("rule", "")).replace("_", " ")
        elif kind == "command_done":
            self.outcome = ev.get("outcome")
            self.latency_ms = ev.get("speech_to_action_ms")
            if self.heard is None and ev.get("text"):
                self.heard = ev.get("text")
            self.phase = "done"
            self.done_at = self.clock()
        elif kind == "verify":
            self.verified = bool(ev.get("ok")) and self.verified is not False
            self.done_at = self.clock()

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
        if self.phase in ("done", "notice"):
            self.phase = "idle"

    def _probe_showing(self) -> bool:
        p = self.probe
        if p is None:
            return False
        return p["ended_at"] is None or self.clock() - p["ended_at"] < PROBE_HOLD_SEC

    def _command_showing(self) -> bool:
        if self.phase in ("listening", "processing"):
            return True
        return (
            self.phase in ("done", "notice")
            and self.done_at is not None
            and self.clock() - self.done_at < HOLD_SEC
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

    def notice(self, text: str) -> None:
        """Show a one-line message (e.g. "Overlay pinned")."""
        if self.phase in ("listening", "processing"):
            return
        self._reset(None)
        self.reply = text
        self.phase = "notice"
        self.done_at = self.clock()

    def cycle_mode(self) -> str:
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        return self.mode

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    @property
    def visible(self) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "pinned":
            return True
        return self._command_showing() or self._probe_showing()

    def lines(self) -> list[Line]:
        # A voice command (live, or just finished) takes the screen; the probe
        # view comes back when it's done.
        if not self._command_showing() and self._probe_showing():
            return self._probe_lines()
        if self.phase == "idle":
            return [Line("VIMAAN", "Hold the push-to-talk key and speak", "muted")]
        if self.phase == "notice":
            return [Line("VIMAAN", self.reply or "", "text")]
        out: list[Line] = []
        if self.phase == "listening":
            out.append(Line("REC", self.partial or "Listening...", "live"))
            return out
        if self.heard is not None:
            meta = []
            if self.backend:
                meta.append(self.backend)
            if self.stt_ms is not None:
                meta.append(f"{self.stt_ms:.0f} ms")
            out.append(Line("HEARD", f'"{self.heard}"', "text", "  ".join(meta)))
        else:
            out.append(Line("...", self.partial or "Recognising...", "muted"))
        if self.intent is not None:
            conf = "" if self.confidence is None else f"{self.confidence:.0%}"
            weak = self.confidence is not None and self.confidence < 0.7
            text = describe_intent(self.intent)
            if self.slots:
                text += "   " + describe_slots(self.slots)
            out.append(Line("MEANT", text, "warn" if weak else "accent", conf))
        if self.phase == "done":
            badge, tone = OUTCOME_BADGE.get(self.outcome or "", ("ERROR", "error"))
            if self.verified is False:
                badge, tone = "NO RESPONSE", "error"
            text = self.reply or (self.outcome or "").replace("_", " ")
            if self.reason:
                text += f"  ({self.reason})"
            meta = []
            if self.verified:
                meta.append("verified")
            if self.latency_ms is not None:
                meta.append(f"{self.latency_ms:.0f} ms")
            out.append(Line(badge, text, tone, "  ".join(meta)))
        return out
