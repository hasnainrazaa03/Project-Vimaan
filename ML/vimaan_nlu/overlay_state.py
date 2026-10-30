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

HOLD_SEC = 6.0  # how long the result stays up after a command finishes
MODES = ("auto", "pinned", "off")  # auto: shows on activity, then fades

# Tones are mapped to colours by the renderer.
OUTCOME_BADGE: dict[str, tuple[str, str]] = {
    "executed": ("DONE", "ok"),
    "no_op": ("NO CHANGE", "ok"),
    "pending": ("WORKING", "accent"),
    "safety_hold": ("CONFIRM?", "warn"),
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
}

# Display names until E0's command table provides them.
INTENT_NAMES: dict[str, str] = {
    "set_autopilot_heading": "Heading",
    "set_autopilot_altitude": "Altitude",
    "set_flight_level": "Flight level",
    "set_com_frequency": "COM frequency",
    "toggle_landing_gear": "Landing gear",
    "toggle_flaps": "Flaps",
    "toggle_autopilot_1": "Autopilot 1",
    "toggle_autopilot_2": "Autopilot 2",
    "toggle_flight_director_1": "Flight director 1",
    "toggle_flight_director_2": "Flight director 2",
    "toggle_parking_brake": "Parking brake",
    "toggle_engine_1": "Engine 1",
    "toggle_engine_2": "Engine 2",
    "None": "Not a command",
    "ask_status_generic": "Status question",
    "ask_time": "Time question",
    "chit_chat_greeting": "Greeting",
}
SLOT_NAMES: dict[str, str] = {
    "degrees": "hdg",
    "altitude": "alt",
    "flight_level": "FL",
    "frequency": "freq",
    "com_port": "COM",
}
REASONS: dict[str, str] = {
    "no_command_word": "no command word heard",
    "one_digit_heading": "heading needs three digits",
}


@dataclass(frozen=True)
class Line:
    badge: str  # short status label, drawn in the tone colour ("" for none)
    text: str
    tone: str  # live | text | muted | accent | ok | warn | error
    detail: str = ""  # right-aligned meta (backend, timings)


def describe_intent(intent: str) -> str:
    return INTENT_NAMES.get(intent, intent.replace("_", " ").capitalize())


def describe_slots(slots: dict[str, Any]) -> str:
    parts = []
    for name, value in slots.items():
        if name == "state":
            parts.append(str(value).upper())
        else:
            parts.append(f"{SLOT_NAMES.get(name, name.replace('_', ' '))} {value}")
    return "  ".join(parts)


class OverlayState:
    def __init__(self, clock: Callable[[], float] = time.monotonic, mode: str = "auto") -> None:
        self.clock = clock
        self.mode = mode if mode in MODES else "auto"
        self._reset(None)
        self.phase = "idle"  # idle | listening | processing | done | notice
        self.done_at: float | None = None

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
        if self.phase in ("listening", "processing"):
            return True
        if self.phase in ("done", "notice") and self.done_at is not None:
            return self.clock() - self.done_at < HOLD_SEC
        return False

    def lines(self) -> list[Line]:
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
