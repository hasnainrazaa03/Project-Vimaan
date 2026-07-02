"""Safety interlocks for irreversible / dangerous commands (Phase 6B).

A small, *pure-Python* layer (no torch, no XPPython3) that decides whether a
predicted command is risky in the current flight context and therefore needs
an explicit confirmation before it executes. The plugin fills a
:class:`SafetyContext` from datarefs and consults :func:`evaluate_safety`; the
:class:`ConfirmationGate` tracks the "armed, waiting for a repeat" state.

Design rules
------------
- **Fail safe.** When a safety-relevant signal is unavailable (dataref missing
  -> ``None``), treat the situation as risky and require confirmation rather
  than silently skipping the check.
- **Only risky-context commands are gated.** Normal operations stay one-press;
  confirmation is requested only when a rule actually fires.
- Keeping the decision logic here (not in the plugin) makes it unit-testable
  without a simulator.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

# Thresholds (documented + tested). Tune in one place.
GEAR_UP_MIN_AGL_FT = 1000.0
AIRBORNE_AGL_FT = 50.0
AIRBORNE_IAS_KT = 40.0
PARKING_BRAKE_MAX_GROUND_SPEED_KT = 5.0

ENGINE_INTENTS = frozenset(
    {"toggle_engine_1", "toggle_engine_2", "toggle_engine_3", "toggle_engine_4"}
)


@dataclass(frozen=True)
class SafetyContext:
    """Flight state the interlocks reason about. Any field may be ``None`` when
    its dataref is unavailable; rules then fail safe."""

    agl_ft: float | None = None
    ground_speed_kt: float | None = None
    ias_kt: float | None = None
    on_ground: bool | None = None


@dataclass(frozen=True)
class SafetyVerdict:
    requires_confirmation: bool
    message: str = ""
    rule: str = ""
    fail_safe: bool = False  # True when triggered by missing data, not a known-risky state


def _airborne(ctx: SafetyContext) -> bool | None:
    """True if airborne, False if on the ground, None if genuinely unknown."""
    signals = []
    if ctx.on_ground is not None:
        signals.append(not ctx.on_ground)
    if ctx.agl_ft is not None:
        signals.append(ctx.agl_ft > AIRBORNE_AGL_FT)
    if ctx.ias_kt is not None:
        signals.append(ctx.ias_kt > AIRBORNE_IAS_KT)
    if not signals:
        return None
    return any(signals)


@dataclass(frozen=True)
class _Rule:
    name: str
    applies_to: Callable[[str, dict], bool]
    # returns True (risky), False (safe), or None (unknown -> fail safe)
    risky_when: Callable[[SafetyContext], bool | None]
    message: str


def _gear_up_low(ctx: SafetyContext) -> bool | None:
    if ctx.agl_ft is None:
        return None
    return ctx.agl_ft < GEAR_UP_MIN_AGL_FT


def _parking_brake_fast(ctx: SafetyContext) -> bool | None:
    if ctx.ground_speed_kt is None:
        return None
    return ctx.ground_speed_kt > PARKING_BRAKE_MAX_GROUND_SPEED_KT


RULES: tuple[_Rule, ...] = (
    _Rule(
        name="gear_up_low_altitude",
        applies_to=lambda i, s: i == "toggle_landing_gear" and s.get("state") == "up",
        risky_when=_gear_up_low,
        message="Confirm gear up, low altitude",
    ),
    _Rule(
        name="engine_shutdown_in_flight",
        applies_to=lambda i, s: i in ENGINE_INTENTS and s.get("state") == "off",
        risky_when=_airborne,
        message="Confirm engine shutdown in flight",
    ),
    _Rule(
        name="parking_brake_at_speed",
        applies_to=lambda i, s: i == "toggle_parking_brake" and s.get("state") == "on",
        risky_when=_parking_brake_fast,
        message="Confirm parking brake at speed",
    ),
)


def evaluate_safety(intent: str, slots: dict, ctx: SafetyContext) -> SafetyVerdict:
    """Return whether ``intent`` (with ``slots``) needs confirmation in ``ctx``."""
    slots = slots or {}
    for rule in RULES:
        if not rule.applies_to(intent, slots):
            continue
        risky = rule.risky_when(ctx)
        if risky is None:
            return SafetyVerdict(True, rule.message, rule.name, fail_safe=True)
        if risky:
            return SafetyVerdict(True, rule.message, rule.name)
    return SafetyVerdict(False)


def command_token(intent: str, slots: dict) -> str:
    """Stable identity for a command so a *repeat* of the same command confirms
    it. Keyed on intent + the action slot (robust to minor STT variation)."""
    slots = slots or {}
    return f"{intent}:{slots.get('state', '')}"


class ConfirmationGate:
    """Tracks a single 'armed, waiting for a repeat' confirmation.

    A risky command arms the gate; repeating the *same* command (same
    :func:`command_token`) within ``window_sec`` confirms and executes it.
    Anything else clears the pending state. ``clock`` is injectable for tests.
    """

    def __init__(self, window_sec: float = 8.0, clock: Callable[[], float] = time.monotonic):
        self._window = window_sec
        self._clock = clock
        self._pending: tuple[str, float] | None = None

    def arm(self, token: str) -> None:
        self._pending = (token, self._clock())

    def pending_token(self) -> str | None:
        if self._pending is None:
            return None
        token, armed_at = self._pending
        if self._clock() - armed_at > self._window:
            self._pending = None
            return None
        return token

    def confirm(self, token: str) -> bool:
        """True if ``token`` matches the live pending command; consumes it."""
        live = self.pending_token()
        self._pending = None
        return live is not None and live == token

    def clear(self) -> None:
        self._pending = None
