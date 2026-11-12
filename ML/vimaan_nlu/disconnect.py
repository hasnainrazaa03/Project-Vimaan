"""Why did the autopilot disconnect?

Flights 203741 (Baron) and the 737 flights lost the autopilot with nothing in
the log to say why: Vimaan hadn't acted for half a minute, and the log had no
stick position or button presses. The plugin now keeps a few seconds of
pilot stick input and every autopilot-off command anyone fires (keyboard,
joystick button, cockpit click, another plugin), and when the autopilot
drops, this module turns them into a cause, a log entry and a callout.

Pure Python, no X-Plane: the plugin feeds it readings.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

# Commands that turn the autopilot off, or hand control back to the pilot.
# All verified in X-Plane 12.4's Commands.txt, and the cockpits that use them
# (the Laminar Baron, King Air, 737 and A330 manipulators).
WATCHED_COMMANDS: tuple[str, ...] = (
    "sim/autopilot/servos_toggle",
    "sim/autopilot/servos2_toggle",
    "sim/autopilot/servos_fdir_off",
    "sim/autopilot/servos_fdir2_off",
    "sim/autopilot/servos_fdir_yawd_off",
    "sim/autopilot/servos_fdir_yawd_trim_off",
    "sim/autopilot/servos_off_any",
    "sim/autopilot/servos_yawd_off_any",
    "sim/autopilot/servos_yawd_trim_off_any",
    "sim/autopilot/fdir_servos_down_one",
    "sim/autopilot/fdir2_servos_down_one",
    "sim/autopilot/fdir_toggle",
    "sim/autopilot/fdir2_toggle",
    "sim/autopilot/electric_master_off",
    "sim/autopilot/control_wheel_steer",
    "sim/autopilot/CWSA",
    "sim/autopilot/CWSB",
    "sim/autopilot/take_off_go_around",
    # Laminar 737: the MCP disengage bar, and CMD A/B (pressed again = off).
    # They exist only while the 737 is loaded.
    "laminar/B738/autopilot/disconnect_toggle",
    "laminar/B738/autopilot/cmd_a_press",
    "laminar/B738/autopilot/cmd_b_press",
)

# Look back this far from when the drop is seen. The state watch runs once a
# second, so the drop itself is up to a second old by then.
LOOKBACK_SEC = 4.0
# Keep this much stick history.
KEEP_SEC = 6.0
# Sample the stick at most this often.
SAMPLE_EVERY_SEC = 0.1
# Pilot stick deflection (ratio of full travel) that counts as "moved".
STICK_MOVED = 0.2
AXES = ("pitch", "roll", "yaw")

# Who disconnected it, in the order they are checked.
CAUSES = ("reload", "probe", "vimaan", "power", "failure", "command", "stick", "unknown")

CALLOUTS: Mapping[str, str | None] = {
    "reload": None,  # a new flight or aircraft
    "probe": None,  # the probe has its own messages
    "vimaan": None,  # Vimaan already said "Autopilot 1 off"
    "power": "Autopilot disconnected. The autopilot has no power.",
    "failure": "Autopilot disconnected. Autopilot failure.",
    "command": "Autopilot disconnected by a switch or button.",
    "stick": "Autopilot disconnected. Stick moved.",
    "unknown": "Autopilot disconnected.",
}


class StickTrace:
    """The last few seconds of pilot stick input."""

    def __init__(self, keep_sec: float = KEEP_SEC, every_sec: float = SAMPLE_EVERY_SEC):
        self.keep_sec = keep_sec
        self.every_sec = every_sec
        self._samples: deque[tuple[float, dict[str, float]]] = deque()

    def due(self, now: float) -> bool:
        return not self._samples or now - self._samples[-1][0] >= self.every_sec

    def add(self, now: float, values: Mapping[str, float | None]) -> None:
        self._samples.append((now, {a: v for a in AXES if (v := values.get(a)) is not None}))
        while self._samples and now - self._samples[0][0] > self.keep_sec:
            self._samples.popleft()

    def peak(self, now: float, lookback: float = LOOKBACK_SEC) -> dict[str, float]:
        """The largest deflection per axis (signed) in the last ``lookback`` s."""
        out: dict[str, float] = {}
        for t, values in self._samples:
            if now - t > lookback:
                continue
            for axis, v in values.items():
                if axis not in out or abs(v) > abs(out[axis]):
                    out[axis] = v
        return {a: round(v, 3) for a, v in out.items()}


class CommandTrace:
    """Recent autopilot-off commands, and whether Vimaan fired them."""

    def __init__(self, keep_sec: float = KEEP_SEC):
        self.keep_sec = keep_sec
        self._seen: deque[tuple[float, str, bool]] = deque()

    def add(self, now: float, name: str, by_vimaan: bool) -> None:
        self._seen.append((now, name, by_vimaan))
        while self._seen and now - self._seen[0][0] > self.keep_sec:
            self._seen.popleft()

    def recent(self, now: float, lookback: float = LOOKBACK_SEC) -> list[dict[str, Any]]:
        return [
            {"command": name, "sec_before": round(now - t, 1), "by_vimaan": by_vimaan}
            for t, name, by_vimaan in self._seen
            if now - t <= lookback
        ]


def dropped(previous: Mapping[str, Any] | None, state: Mapping[str, Any]) -> bool:
    """Did the autopilot go from on to off?"""
    if not previous:
        return False
    return any(previous.get(k) == 1 and state.get(k) == 0 for k in ("servos_on", "autopilot_on"))


def diagnose(
    *,
    stick_peak: Mapping[str, float],
    commands: Iterable[Mapping[str, Any]],
    has_power: int | None,
    fail_warn: int | None,
    vimaan_turned_off: bool = False,
    probe_running: bool = False,
    reloading: bool = False,
) -> dict[str, Any]:
    """The most likely cause, what to show, and what to say (None: silent).

    ``commands`` are the recent autopilot-off commands (``CommandTrace``);
    ``vimaan_turned_off`` is true when the pilot asked Vimaan for it."""
    commands = list(commands)
    others = [c for c in commands if not c.get("by_vimaan")]
    moved = {a: v for a, v in stick_peak.items() if abs(v) >= STICK_MOVED}
    detail = []
    if others:
        detail.append("command " + ", ".join(c["command"] for c in others))
    if moved:
        detail.append("stick " + ", ".join(f"{a} {v:+.2f}" for a, v in moved.items()))
    if reloading:
        cause = "reload"
    elif probe_running:
        cause = "probe"
    elif vimaan_turned_off or (any(c.get("by_vimaan") for c in commands) and not others):
        cause = "vimaan"
    elif has_power == 0:
        cause = "power"
        detail.insert(0, "no autopilot power")
    elif fail_warn == 1:
        cause = "failure"
        detail.insert(0, "autopilot fail warning")
    elif others:
        cause = "command"
    elif moved:
        cause = "stick"
    else:
        cause = "unknown"
    return {"cause": cause, "detail": "; ".join(detail) or None, "say": CALLOUTS[cause]}
