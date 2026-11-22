"""Phase of flight (Phase F.1), from a sample of the aircraft once a second.

Pure Python (no XPPython3): the plugin fills a :class:`FlightSample` from
datarefs and feeds it to a :class:`PhaseTracker`; recorded samples can be
replayed through a fresh tracker (:func:`replay`) to check or tune it without
the sim.

Observe only: nothing acts on the phase yet. Safety rules, callouts and
answers that read it come later, each with the owner's approval.

Phases
------
On the ground: ``parked`` (not moving), ``taxi``, ``takeoff_roll`` (fast,
not after a landing), ``rollout`` (fast, after a landing). In the air:
``initial_climb`` (from liftoff to 1,000 ft above the ground), ``climb``,
``cruise`` (level), ``descent``, ``approach`` (low with gear or flaps out,
or low and still coming down from a descent), ``landing`` (the last 100 ft,
after an approach),
``go_around`` (climbing away from an approach or a landing below 1,000 ft
above the ground, until 1,000 ft; higher up it is a climb; configured and
coming down again below 1,000 ft, it was a balloon: back to the approach, or
the landing below 100 ft). ``unknown`` when neither on-ground nor height can be read.

A new phase must hold for a few seconds before it is taken (vertical speed
wanders around a level-off), except the ground/air change, which takes one
second; a bounce after landing (airborne again, below 50 ft) stays the
rollout.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

PHASES = (
    "unknown",
    "parked",
    "taxi",
    "takeoff_roll",
    "initial_climb",
    "climb",
    "cruise",
    "descent",
    "approach",
    "landing",
    "go_around",
    "rollout",
)
GROUND_PHASES = frozenset({"parked", "taxi", "takeoff_roll", "rollout"})

# Thresholds (documented + tested). Tune in one place.
STOPPED_KT = 1.0  # on the ground, slower than this: not moving
ROLL_KT = 40.0  # on the ground, faster than this: a takeoff or landing roll
LEVEL_FPM = 300.0  # vertical speed within +-this: level (cruise)
INITIAL_CLIMB_AGL_FT = 1000.0  # after liftoff or a go-around, until this height
APPROACH_AGL_FT = 2500.0  # below this, configured or descending from a descent: approach
LANDING_AGL_FT = 100.0  # below this, after an approach, not climbing: landing
GO_AROUND_FPM = 500.0  # climbing at least this fast from an approach: go-around
GEAR_DOWN_RATIO = 0.99  # gear deploy ratio at or above this: down
FLAPS_OUT_RATIO = 0.01  # flap handle ratio above this: flaps out
ON_GROUND_AGL_FT = 5.0  # with no on-ground signal, this low counts as on the ground
BOUNCE_AGL_FT = 50.0  # airborne again after a landing, below this: a bounce, still rolling out
DWELL_SEC = 3.0  # a new phase must hold this long ...
GROUND_DWELL_SEC = 1.0  # ... or this long when it changes ground/air


@dataclass(frozen=True)
class FlightSample:
    """The aircraft at one moment. Any field may be ``None`` when its
    dataref is unavailable."""

    t: float  # seconds, any clock that only moves forward
    on_ground: bool | None = None
    agl_ft: float | None = None
    gs_kt: float | None = None  # ground speed
    ias_kt: float | None = None
    vs_fpm: float | None = None  # + is up
    gear: float | None = None  # deploy ratio, 0 up .. 1 down (fixed gear: 1)
    flaps: float | None = None  # flap handle ratio, 0 up .. 1 full
    altitude_ft: float | None = None  # indicated, for the log only


def _on_ground(s: FlightSample) -> bool | None:
    if s.on_ground is not None:
        return s.on_ground
    if s.agl_ft is not None:
        return s.agl_ft < ON_GROUND_AGL_FT
    return None


def _configured(s: FlightSample) -> bool:
    """Gear down (retractable gear) or flaps out: set up to land."""
    return (s.flaps is not None and s.flaps > FLAPS_OUT_RATIO) or (
        s.gear is not None and s.gear >= GEAR_DOWN_RATIO
    )


def candidate(s: FlightSample, previous: str | None) -> str:
    """The phase this sample shows, given the phase so far (``None`` at the
    start of a flight: an air start begins wherever it is, not in a climb
    after takeoff)."""
    ground = _on_ground(s)
    if ground is None:
        return "unknown"
    gs = s.gs_kt or 0.0
    if ground:
        if gs >= ROLL_KT:
            # Fast on the ground after flying is a landing roll; before, a takeoff.
            if previous in GROUND_PHASES or previous in (None, "unknown"):
                return "rollout" if previous == "rollout" else "takeoff_roll"
            return "rollout"
        return "parked" if gs < STOPPED_KT else "taxi"

    agl = s.agl_ft
    vs = s.vs_fpm if s.vs_fpm is not None else 0.0
    if previous == "rollout" and agl is not None and agl < BOUNCE_AGL_FT:
        return "rollout"  # a bounce, not a touch-and-go
    if previous in GROUND_PHASES:  # just lifted off (or a touch-and-go)
        return "initial_climb"
    if previous == "go_around" and (agl is None or agl < INITIAL_CLIMB_AGL_FT):
        # A go-around climbs. Gear or flaps out and coming down again, it was
        # a balloon on the approach: the MD-82's flaps 28 at 172 kt pitched it
        # up at +1,100 fpm at 926 ft, then it dived back to the runway, and
        # the go-around held to touchdown, three minutes, while the approach
        # rules ("gear up" held) were off (L.2.3 flights, 2026-11-22).
        if agl is not None and vs <= -LEVEL_FPM and _configured(s):
            return "landing" if agl < LANDING_AGL_FT else "approach"
        return previous
    if previous == "initial_climb" and (agl is None or agl < INITIAL_CLIMB_AGL_FT):
        return previous
    # Below 1,000 ft only: higher up, a climb from an approach is a climb (the
    # go-around ends at 1,000 ft anyway). X-Plane's final-approach start at
    # 1,500 ft, levelled off by the autopilot at +700 fpm, read as three
    # go-arounds (Phase F.3 replay, 2026-11-19).
    if (
        previous in ("approach", "landing")
        and vs >= GO_AROUND_FPM
        and (agl is None or agl < INITIAL_CLIMB_AGL_FT)
    ):
        return "go_around"
    if agl is not None and agl < APPROACH_AGL_FT and vs < LEVEL_FPM:
        if agl < LANDING_AGL_FT and previous in ("approach", "landing"):
            return "landing"
        # Clean, a descent only from a descent: a dip in level flight over
        # hills isn't an approach (the Baron in altitude hold at 2,400 ft
        # above the ground, Tier 1 replay 2026-11-19); it's a descent first.
        descending = previous == "descent" and vs <= -LEVEL_FPM
        if descending or _configured(s) or previous in ("approach", "landing"):
            return "approach"
    if vs >= LEVEL_FPM:
        return "climb"
    if vs <= -LEVEL_FPM:
        return "descent"
    return "cruise"


class PhaseTracker:
    """Feed it samples in time order; :meth:`update` returns the new phase
    when it changes, else ``None``. :attr:`phase` is the phase now."""

    def __init__(self, dwell_sec: float = DWELL_SEC, ground_dwell_sec: float = GROUND_DWELL_SEC):
        self.dwell_sec = dwell_sec
        self.ground_dwell_sec = ground_dwell_sec
        self.reset()

    def reset(self) -> None:
        """A new flight (an aircraft or airport loaded): start again."""
        self.phase = "unknown"
        self.since: float | None = None  # when the phase was taken
        self._started = False
        self._pending: str | None = None
        self._pending_since = 0.0

    def update(self, s: FlightSample) -> str | None:
        want = candidate(s, self.phase if self._started else None)
        if not self._started and want != "unknown":
            self._started = True
            return self._take(want, s.t)
        if want == self.phase:
            self._pending = None
            return None
        if want != self._pending:
            self._pending, self._pending_since = want, s.t
        changes_ground = (want in GROUND_PHASES) != (self.phase in GROUND_PHASES)
        dwell = self.ground_dwell_sec if changes_ground else self.dwell_sec
        if want == "unknown" or s.t - self._pending_since >= dwell:
            return self._take(want, self._pending_since)
        return None

    def _take(self, phase: str, t: float) -> str:
        self.phase, self.since, self._pending = phase, t, None
        return phase


def replay(
    samples: Iterable[FlightSample], tracker: PhaseTracker | None = None
) -> list[tuple[float, str]]:
    """(time, phase) at every change, for recorded samples."""
    tracker = tracker or PhaseTracker()
    changes = []
    for s in samples:
        if tracker.update(s) is not None:
            changes.append((s.t if tracker.since is None else tracker.since, tracker.phase))
    return changes


SAMPLE_FIELDS = ("on_ground", "agl_ft", "gs_kt", "ias_kt", "vs_fpm", "gear", "flaps", "altitude_ft")


def sample_from_event(ev: dict) -> FlightSample:
    """A recorded ``flight`` event back into a sample (scripts, tests)."""
    return FlightSample(t=float(ev["t"]), **{k: ev.get(k) for k in SAMPLE_FIELDS})
