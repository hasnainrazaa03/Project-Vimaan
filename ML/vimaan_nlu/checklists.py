"""Checklists, read out and checked against the sim (phase L.4).

Pure Python: the plugin reads each item's state from the sim and says what
:func:`check` returns. The lists, the checks, the words and when a checklist
is suggested live here, so they are tested without X-Plane.

The owner's decisions (2026-11-22; docs/PHASE_PLAN.md, "L.4 Checklists"):

1. Vimaan reads each item, checks it and says the result ("Gear down,
   checked"), moving on by itself, then sums up what is open.
2. A wrong item is offered ("Set landing lights on? Say confirm, or
   cancel."). "Confirm" sets it through the command's own handler and the
   safety rules. Some items are only checked (the flaps for landing).
3. A checklist starts when the pilot asks. It is also suggested at a phase
   change, opt-in like the callouts (:class:`Suggestions`).

An item the aircraft lacks is skipped (fixed gear, no auto brake, no yaw
damper, no speed brakes that arm). The plugin decides that the way the
item's own command does, so a skipped item is one whose command would say
"This aircraft has no ...".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from vimaan_nlu.flight_phase import GEAR_DOWN_RATIO

NAMES = (
    "before takeoff",
    "after takeoff",
    "approach",
    "before landing",
    "after landing",
    "shutdown",
)

# What the plugin reads for an item (PythonInterface._checklist_read).
READS = frozenset(
    {
        "flaps",  # "up", "first" (the first notch) or "out" (further)
        "gear",  # "up", "down" or "moving"
        "switch",  # a switch's setting (vimaan_nlu.switches), "mixed" when they differ
        "transponder",  # "off", "standby", "on" or "alt"
        "auto_brake",  # "off", "rto", "1", "2", "3" or "max"
        "speed_brake",  # "retract", "armed", "half" or "extend"
        "autopilot",  # "on" or "off"
        "parking_brake",  # "set" or "released"
        "engines",  # "off" (none running) or "running"
    }
)

# Pacing: Vimaan moves on once an item is said. X-Plane speaks about 150
# words a minute; a little more is allowed so that no item is cut short.
WORD_SEC = 0.42
GAP_SEC = 0.9
# How long an offer waits for "confirm" or "cancel" once it is said: the
# confirmation window of a safety hold (PythonInterface.CONFIRM_WINDOW_SEC).
OFFER_WAIT_SEC = 12.0
OFFER_TAIL = "Say confirm, or cancel."


@dataclass(frozen=True)
class Item:
    """One line of a checklist.

    ``name``: what it checks, as the summary says it ("landing lights").
    ``reads``: what the plugin reads (``READS``). ``keys``: the switches, for
    ``"switch"``. ``want``: the states that pass. ``offer``: the command that
    sets it (intent, slots), or None when it is only checked; then ``hint``
    says what is wrong. ``ask``: the offer, as asked ("Set landing lights on").
    ``only``: "jet" or "prop" when it is for one kind of aircraft.
    ``after_engines``: right only once the engines are off (the beacon).
    """

    name: str
    reads: str
    want: frozenset[str]
    keys: tuple[str, ...] = ()
    offer: tuple[str, Mapping[str, Any]] | None = None
    ask: str = ""
    hint: str = ""
    only: str = ""
    after_engines: bool = False

    def __post_init__(self) -> None:
        if self.reads not in READS:
            raise ValueError(f"unknown reading {self.reads!r}")
        if self.offer is not None:
            object.__setattr__(
                self, "offer", (self.offer[0], MappingProxyType(dict(self.offer[1])))
            )

    @property
    def label(self) -> str:
        """The name at the start of what is said ("Landing lights")."""
        return self.name[:1].upper() + self.name[1:]


@dataclass(frozen=True)
class Result:
    """What :func:`check` found: ``ok``, what to say, and the command to
    offer (intent, slots) when the item is wrong and can be set."""

    ok: bool
    said: str
    offer: tuple[str, Mapping[str, Any]] | None = None


def _want(*states: str) -> frozenset[str]:
    return frozenset(states)


def _light(light: str, name: str, state: str, **kw: Any) -> Item:
    return Item(
        name,
        "switch",
        _want(state),
        keys=(f"light:{light}",),
        offer=("toggle_lights", {"light": light, "switch": state}),
        ask=f"Set {name} {state}",
        **kw,
    )


def _seat_belts(state: str) -> Item:
    return Item(
        "seat belt sign",
        "switch",
        _want(state),
        keys=("sign:seat belt",),
        offer=("toggle_signs", {"sign": "seat belt", "switch": state}),
        ask=f"Set the seat belt sign {state}",
    )


def _fuel_pumps(state: str) -> Item:
    # A pump left on "auto" is on (fuel_pump_on: 0 off, 1 on, 2 auto).
    return Item(
        "fuel pumps",
        "switch",
        _want(state, "auto") if state == "on" else _want(state),
        keys=("fuel:left", "fuel:center", "fuel:right"),
        offer=("toggle_fuel_pump", {"switch": state}),
        ask=f"Set fuel pumps {state}",
    )


def _speed_brakes(state: str, **kw: Any) -> Item:
    return Item(
        "speed brakes",
        "speed_brake",
        _want(state),
        offer=("set_speed_brake", {"speedbrake": state}),
        ask="Retract the speed brakes" if state == "retract" else "Arm the speed brakes",
        **kw,
    )


def _flaps_up() -> Item:
    # The "flaps up when slow" hold still applies: the offer goes through the
    # safety rules like any command.
    return Item(
        "flaps", "flaps", _want("up"), offer=("set_flaps", {"position": "up"}), ask="Set flaps up"
    )


def _auto_brake_off() -> Item:
    return Item(
        "auto brake",
        "auto_brake",
        _want("off"),
        offer=("set_autobrake", {"autobrake": "off"}),
        ask="Set auto brake off",
    )


LANDING_LIGHTS_ON = _light("landing", "landing lights", "on")

# The approved lists (2026-11-22), in the order they are read.
CHECKLISTS: Mapping[str, tuple[Item, ...]] = MappingProxyType(
    {
        "before takeoff": (
            # Jets: flaps out; when up, the first notch is offered ("flaps
            # down" is one notch). Props: up or the first notch, checked only.
            Item(
                "flaps",
                "flaps",
                _want("first", "out"),
                offer=("set_flaps", {"position": "down"}),
                ask="Set the first notch of flaps",
                only="jet",
            ),
            Item("flaps", "flaps", _want("up", "first"), hint="Not set for takeoff", only="prop"),
            Item(
                "transponder",
                "transponder",
                _want("alt"),
                offer=("set_transponder_mode", {"xpdr_mode": "alt"}),
                ask="Set the transponder to altitude",
            ),
            LANDING_LIGHTS_ON,
            _light("strobe", "strobes", "on"),
            # Set at all passes (RTO, or a level on a jet without RTO); when
            # off, RTO is offered where the aircraft has it.
            Item(
                "auto brake",
                "auto_brake",
                _want("rto", "1", "2", "3", "max"),
                offer=("set_autobrake", {"autobrake": "r t o"}),
                ask="Set auto brake R T O",
                only="jet",
            ),
            _speed_brakes("retract"),
            _fuel_pumps("on"),
            _seat_belts("on"),
            Item(
                "yaw damper",
                "switch",
                _want("on"),
                keys=("yaw_damper",),
                offer=("toggle_yaw_damper", {"state": "on"}),
                ask="Set the yaw damper on",
            ),
            Item(
                "autopilot",
                "autopilot",
                _want("off"),
                offer=("toggle_autopilot", {"state": "off"}),
                ask="Set the autopilot off",
            ),
        ),
        "after takeoff": (
            Item(
                "gear",
                "gear",
                _want("up"),
                offer=("toggle_landing_gear", {"state": "up"}),
                ask="Raise the gear",
            ),
            _flaps_up(),
            _auto_brake_off(),
        ),
        "approach": (
            _seat_belts("on"),
            LANDING_LIGHTS_ON,
            # Jets: set for landing; when off (or RTO), the lowest setting.
            Item(
                "auto brake",
                "auto_brake",
                _want("1", "2", "3", "max"),
                offer=("set_autobrake", {"autobrake_level": 1}),
                ask="Set auto brake one",  # the plugin says the A330's "low"
                only="jet",
            ),
        ),
        "before landing": (
            Item(
                "gear",
                "gear",
                _want("down"),
                offer=("toggle_landing_gear", {"state": "down"}),
                ask="Lower the gear",
            ),
            Item("flaps", "flaps", _want("first", "out"), hint="Not set for landing"),
            _speed_brakes("armed", only="jet"),  # jets whose speed brakes arm
            LANDING_LIGHTS_ON,
        ),
        "after landing": (
            _speed_brakes("retract"),
            _flaps_up(),
            _auto_brake_off(),
            _light("strobe", "strobes", "off"),
            _light("landing", "landing lights", "off"),
            _light("taxi", "taxi lights", "on"),
        ),
        "shutdown": (
            Item(
                "parking brake",
                "parking_brake",
                _want("set"),
                offer=("toggle_parking_brake", {"state": "on"}),
                ask="Set the parking brake",
            ),
            Item(
                "engines",
                "engines",
                _want("off"),
                offer=("toggle_engine", {"state": "off", "instance": "both"}),
                ask="Shut down the engines",
            ),
            _light("beacon", "beacon", "off", after_engines=True),
            _fuel_pumps("off"),
            _seat_belts("off"),
            _light("taxi", "taxi lights", "off"),
            _light("landing", "landing lights", "off"),
        ),
    }
)


def title(name: str) -> str:
    """What is said when it starts ("Before takeoff checklist")."""
    return f"{name[:1].upper()}{name[1:]} checklist"


def speaking_sec(text: str) -> float:
    """About how long X-Plane takes to say ``text``, with a pause after."""
    return GAP_SEC + WORD_SEC * len(text.split())


def check(item: Item, state: str | None, words: str, engines_running: bool = False) -> Result:
    """The item read as ``state`` (None: unreadable), said as ``words``
    ("Landing lights off"). A wrong item with a command is offered."""
    if state is None:
        return Result(False, f"I can't read the {item.name}")
    if state in item.want:
        return Result(True, f"{words}, checked")
    if item.after_engines and engines_running:
        return Result(False, f"{words}, until the engines are off")
    if item.offer is not None:
        return Result(False, offer_words(words, item.ask), item.offer)
    return Result(False, f"{words}. {item.hint}" if item.hint else words)


def offer_words(words: str, ask: str) -> str:
    return f"{words}. {ask}? {OFFER_TAIL}"


def _spoken_list(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def summary(name: str, open_items: Iterable[str]) -> str:
    """What is said at the end: complete, or what is left open."""
    left = list(dict.fromkeys(open_items))  # the order read, once each
    if not left:
        return f"{title(name)} complete"
    count = "one item" if len(left) == 1 else f"{len(left)} items"
    return f"{title(name)} done, {count} open: {_spoken_list(left)}"


def suggestion_words(name: str) -> str:
    return f"{title(name)}? {OFFER_TAIL}"


# When each is suggested (the approved table): the phase change that brings it
# round, once until the flight comes round to it again.
SUGGESTED_AT = MappingProxyType(
    {
        "before takeoff": "starting to taxi after an engine start",
        "after takeoff": "climbing through 1,000 ft",
        "approach": "entering the approach phase",
        "before landing": "gear down on the approach",
        "after landing": "slowing to taxi speed after landing",
        "shutdown": "parked after landing, engines running",
    }
)
# Brought round again by a takeoff or a go-around: the checklists after it.
_AIRBORNE_AGAIN = frozenset(
    {"after takeoff", "approach", "before landing", "after landing", "shutdown"}
)


@dataclass
class Suggestions:
    """Which checklist to suggest at a phase change, if any (opt-in).

    The plugin calls :meth:`update` with each flight sample's phase (from
    vimaan_nlu.flight_phase), whether an engine is running and the gear
    ratio (None for fixed gear). A checklist is suggested at most once until
    the flight comes round to it again (a takeoff, a go-around, a landing or
    an engine start), and not after the pilot has run it."""

    said: set[str] = field(default_factory=set)
    landed: bool = False
    _phase: str | None = None
    _running: bool | None = None
    _gear: float | None = None

    def reset(self) -> None:
        """A new flight: nothing said yet."""
        self.said.clear()
        self.landed = False
        self._phase = self._running = self._gear = None

    def ran(self, name: str) -> None:
        """The pilot ran it (asked, or took the suggestion)."""
        self.said.add(name)

    def update(
        self, phase: str | None, engines_running: bool | None, gear: float | None
    ) -> str | None:
        previous, self._phase = self._phase, phase
        was_running, self._running = self._running, engines_running
        gear_before, self._gear = self._gear, gear
        if was_running is False and engines_running:
            # An engine start: the next taxi is the next departure.
            self.said.discard("before takeoff")
            self.landed = False
        if phase != previous:
            if phase in ("initial_climb", "go_around"):
                self.said -= _AIRBORNE_AGAIN
                self.landed = False
            elif phase == "rollout":
                self.landed = True
                self.said.discard("before takeoff")
        name = self._due(previous, phase, engines_running, gear_before, gear)
        if name is None or name in self.said:
            return None
        self.said.add(name)
        return name

    def _due(self, previous, phase, running, gear_before, gear) -> str | None:
        # Only a phase the flight was seen coming into: not the first one
        # known after a load or an air start (on final, no "approach").
        if phase != previous and previous not in (None, "unknown"):
            if previous == "parked" and phase == "taxi" and running and not self.landed:
                return "before takeoff"
            if previous == "initial_climb" and phase in ("climb", "cruise", "descent"):
                return "after takeoff"
            if phase == "approach" and previous != "go_around":
                return "approach"
            if previous == "rollout" and phase == "taxi":
                return "after landing"
            if previous == "taxi" and phase == "parked" and self.landed and running:
                return "shutdown"
        if (
            phase == "approach"
            and gear is not None
            and gear_before is not None
            and gear_before < GEAR_DOWN_RATIO
            and gear >= GEAR_DOWN_RATIO
        ):
            return "before landing"
        return None
