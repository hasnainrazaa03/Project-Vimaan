"""Spoken callouts from the phase of flight and the speed margins (Phase
F.4). Opt-in: off unless the pilot switches them on (``VIMAAN_CALLOUTS=1``,
or the ``vimaan/toggle_callouts`` command, which a key can be given to).

Pure Python: the plugin calls :meth:`Callouts.update` with each
once-a-second flight sample (the phase from vimaan_nlu.flight_phase, the
margins from vimaan_nlu.envelope) and says what it returns.

The callouts, most urgent first (one at a time: the next waits a second):

- "Speed": within 5 kt of the stall speed (the .acf's, for one weight).
- "Overspeed": beyond Vne (Vmo on jets).
- "Mach": beyond Mmo.
- "Gear speed": beyond Vle with the gear not up.
- "Flap speed": within 5 kt of the flap limit for the notch set, or beyond.
- "Gear not down": on an approach below 500 ft with the gear not down.

Each is said at most once every 30 s while it holds, and none is said on the
ground or in the last 100 ft (the landing phase, or the flare's slow speed
would be a "Speed"). On the flights recorded in X-Plane on 2026-11-19 they
fired only where something was wrong (an over-rotated Baron at 88 kt; the
Baron that slowed to its stall on the approach, 8 s before its autopilot let
go at 81 kt; the A330 through its flap limit; the King Air over Vne) and never on the
clean runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vimaan_nlu.flight_phase import GEAR_DOWN_RATIO, GROUND_PHASES, LANDING_AGL_FT

STALL_MARGIN_KT = 5.0
FLAP_MARGIN_KT = 5.0
GEAR_CALL_AGL_FT = 500.0
REPEAT_SEC = 30.0
# No callouts in these phases: on the ground, the last 100 ft, or not known.
QUIET_PHASES = GROUND_PHASES | {"landing", "unknown"}

# (key, what is said), most urgent first.
CALLOUTS = (
    ("speed", "Speed"),
    ("overspeed", "Overspeed"),
    ("mach", "Mach"),
    ("gear_speed", "Gear speed"),
    ("flap_speed", "Flap speed"),
    ("gear", "Gear not down"),
)
SAID = dict(CALLOUTS)


def _below(margins: dict, name: str, threshold: float) -> bool:
    value = margins.get(name)
    return value is not None and value < threshold


def due(
    phase: str | None,
    margins: dict,
    on_ground: bool | None,
    agl_ft: float | None,
    gear: float | None,
) -> list[str]:
    """The callouts whose condition holds now, most urgent first (keys)."""
    if phase is None or phase in QUIET_PHASES or on_ground is not False:
        return []
    if agl_ft is None or agl_ft < LANDING_AGL_FT:
        return []
    holds = {
        "speed": _below(margins, "stall", STALL_MARGIN_KT),
        "overspeed": _below(margins, "vne", 0.0),
        "mach": _below(margins, "mmo", 0.0),
        "gear_speed": _below(margins, "gear", 0.0),
        "flap_speed": _below(margins, "flaps", FLAP_MARGIN_KT),
        "gear": phase == "approach"
        and agl_ft < GEAR_CALL_AGL_FT
        and gear is not None
        and gear < GEAR_DOWN_RATIO,
    }
    return [key for key, _ in CALLOUTS if holds[key]]


@dataclass
class Callouts:
    """Which callout to say now, if any, rate-limited per callout."""

    repeat_sec: float = REPEAT_SEC
    last: dict[str, float] = field(default_factory=dict)  # key -> when it was said

    def reset(self) -> None:
        """A new flight: nothing said yet."""
        self.last.clear()

    def update(
        self,
        t: float,
        phase: str | None,
        margins: dict,
        on_ground: bool | None,
        agl_ft: float | None,
        gear: float | None,
    ) -> str | None:
        """The key of the callout to say now (the most urgent one due and not
        said in the last 30 s), or None."""
        for key in due(phase, margins, on_ground, agl_ft, gear):
            said = self.last.get(key)
            if said is None or t - said >= self.repeat_sec:
                self.last[key] = t
                return key
        return None
