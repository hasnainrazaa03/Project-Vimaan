"""Relative commands (phase L.1): "turn left twenty", "climb another
thousand", "speed up ten knots", and compass headings ("turn north").

The model names the change (which way, how much). The plugin works out the
target from the aircraft with these functions and reads the result back, so
the pilot always hears the value that was set ("Left twenty, heading two five
zero").

Some words don't say whether a number is the target or the change: at 10,000
ft "descend two thousand feet" can be the altitude 2,000 or 2,000 lower, and
v26 set 2,000. When both readings are valid and go the way the pilot said,
the plugin asks, offering both, as it does for "to" and "two"
(``vimaan_nlu.ambiguity``). The owner chose that on 2026-11-20.

Pure Python: no X-Plane, no torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vimaan_nlu.commands import SLOT_VALIDATORS
from vimaan_nlu.normalization import _MAGNITUDE_WORDS, _TEEN_WORDS, _TENS_WORDS, _UNIT_WORDS

COMPASS: Mapping[str, int] = {"north": 360, "east": 90, "south": 180, "west": 270}

# A turn of 180 or more: the autopilot turns the short way, so it could go
# the other way. "Turn right two seven zero" is a heading.
TURN_MAX = 179
# Larger relative altitudes aren't said; Vosk hears "eight" as "a" ("a
# thousand"), so the limit stays below 8,000.
ALTITUDE_CHANGE_MAX = 5000
ALTITUDE_STEP = 100
SPEED_CHANGE_MAX = 50
SPEED_STEP = 5
# Climbing or descending this fast, the autopilot is on its way to a target.
MOVING_FPM = 100.0

# Words that make a number the target ("climb to", "descend and maintain",
# "turn left heading") or the change ("another", "by", "more").
TARGET_WORDS = frozenset({"to", "maintain", "altitude", "level", "at", "heading"})
CHANGE_WORDS = frozenset(
    {
        "another",
        "more",
        "by",
        "further",
        "extra",
        "higher",
        "lower",
        "up",
        "down",
        "from",
        "degrees",
        "degree",
    }
)
SIDES = ("left", "right")
# The commands whose number is a change ("climb another thousand").
CHANGE_INTENTS = frozenset({"change_altitude_by", "change_speed_by"})


def _words(text: str) -> list[str]:
    return text.lower().split()


def says_change(text: str) -> bool:
    """A word that makes a number a change ("another thousand", "up a
    thousand"): there "a" and "another" count as one."""
    return not CHANGE_WORDS.isdisjoint(_words(text))


# Words that make a number a change and never the altitude ("another two
# thousand", "a thousand feet higher"). Not "up"/"down": "climb up to eight
# thousand" is the altitude.
ALWAYS_CHANGE = frozenset({"another", "more", "by", "further", "extra", "higher", "lower"})


def change_said_as_altitude(intent: str, text: str) -> bool:
    """An altitude command whose words say a change: "climb another two
    thousand", "come down two thousand feet". A model without the change
    command (v26) reads those as the altitude and flew 2,000 ft: of data
    v19's altitude-change phrasings, 1,904 acted so (2026-11-20). Refused,
    with the owner's OK the same day. "Up" or "down" counts only with no word
    making the number the target ("climb up to eight thousand")."""
    if intent != "set_autopilot_altitude":
        return False
    words = set(_words(text))
    if not words.isdisjoint(ALWAYS_CHANGE):
        return True
    return not words.isdisjoint({"up", "down"}) and words.isdisjoint(TARGET_WORDS)


def _to_a_number(words: list[str]) -> bool:
    """ "to" right before a number, in digits or in words ("turn left to one
    five zero")."""
    return any(
        w == "to"
        and words[i + 1 : i + 2]
        and (words[i + 1].isdigit() or words[i + 1] in _NUMBER_WORDS)
        for i, w in enumerate(words)
    )


def change_said_as_heading(intent: str, text: str) -> bool:
    """A heading command whose words say a turn: "forty five degrees to the
    right", "turn right another one two nine degrees". A model without the
    turn command (v26) set those as the heading: 708 of data v19's turn
    phrasings acted so (2026-11-20). Refused, with the owner's OK the same
    day. A change word, or a side with "degrees"; not when "heading", or "to"
    before the number, makes it the target ("turn left heading two seven
    zero", "turn left to one five zero degrees")."""
    if intent != "set_autopilot_heading":
        return False
    words = _words(text)
    if "heading" in words or _to_a_number(words):
        return False
    if not ALWAYS_CHANGE.isdisjoint(words):
        return True
    return side_said(text) is not None and not {"degrees", "degree"}.isdisjoint(words)


# A turn smaller than this needs a word that says it is a turn. Of single
# words that aren't commands, Speech Commands' "forward" came out "four
# right" through the command grammar, and v27 turned right 4 degrees (2 of
# 300 clips, 2026-11-20). The owner chose on 2026-11-21 that a bare side and
# number of under 10 degrees is refused; "left twenty" still turns.
SMALL_TURN = 10
TURN_WORDS = frozenset({"turn", "come", "degrees", "degree"})


def small_turn_unsaid(intent: str, slots: Mapping[str, Any], text: str) -> bool:
    """A turn of under ``SMALL_TURN`` degrees with none of ``TURN_WORDS``
    ("four right", "left five"). "Turn right four", "come left five" and
    "right four degrees" still turn."""
    if intent != "turn_by":
        return False
    n = _number(slots.get("turn_degrees"))
    return n is not None and n < SMALL_TURN and TURN_WORDS.isdisjoint(_words(text))


def bare_climb(text: str) -> str | None:
    """ "up" or "down" when a climb or descent is said with no word saying
    whether its number is the altitude or the change ("descend two thousand
    feet"); None otherwise."""
    words = set(_words(text))
    if not words.isdisjoint(TARGET_WORDS) or not words.isdisjoint(CHANGE_WORDS):
        return None
    climb, descend = "climb" in words, "descend" in words
    if climb == descend:
        return None
    return "up" if climb else "down"


def side_said(text: str) -> str | None:
    """ "left" or "right" when exactly one is said. "Right away" names no
    side: v27's first candidate read "turn left to one forty eight right
    away" as a turn to the right."""
    words = _words(text)
    said = {
        w
        for i, w in enumerate(words)
        if w in SIDES and not (w == "right" and words[i + 1 : i + 2] == ["away"])
    }
    return said.pop() if len(said) == 1 else None


def says_heading(text: str) -> bool:
    """The number is a heading: "heading" is said, or "to" comes right before
    the number ("turn left to 15"; the text is normalized, numbers in
    digits). "Fifteen degrees to the left" names the side, not a heading."""
    words = _words(text)
    if "heading" in words:
        return True
    return any(
        w == "to" and words[i + 1 : i + 2] and words[i + 1].isdigit() for i, w in enumerate(words)
    )


def bare_turn(text: str) -> bool:
    """A turn left or right with no word saying whether its number is the
    heading or the change ("turn right one two zero")."""
    words = set(_words(text))
    return (
        side_said(text) is not None
        and words.isdisjoint(TARGET_WORDS)
        and words.isdisjoint(CHANGE_WORDS)
    )


_NUMBER_WORDS = frozenset(_UNIT_WORDS) | frozenset(_TEEN_WORDS) | frozenset(_TENS_WORDS)
_NUMBER_WORDS |= frozenset(_MAGNITUDE_WORDS)


def compass_conflict(text: str) -> bool:
    """A compass heading said with a number or a side ("turn west twenty",
    "turn left south"): there the direction may be a misheard turn. Through
    the command grammar, Speech Commands' "left" came out "west" and "five"
    and "seven" "south" (1 in 300 each, 2026-11-20)."""
    return any(w in _NUMBER_WORDS or w in SIDES or w.isdigit() for w in _words(text))


def _number(value: Any) -> int | None:
    s = str(value or "").strip()
    return int(s) if s.isdigit() else None


def turn_reading(intent: str, slots: Mapping[str, Any], text: str) -> tuple[str, Any] | None:
    """How a turn reads, or None when it's not one this module decides:

    - ("heading", degrees): the number can't be a turn (180 or more, or said
      with a leading zero: "turn left zero two zero" is heading 020);
    - ("turn", (side, degrees)): a turn by that much;
    - ("either", (side, degrees)): 100 to 179 with nothing saying which, so
      ask ("turn right one two zero": heading 120, or 120 degrees right).
    """
    if intent == "turn_by":
        raw = str(slots.get("turn_degrees") or "").strip()
        n, side = _number(raw), slots.get("turn")
        if n is None or side not in SIDES:
            return None
        if len(raw) == 3 and raw.startswith("0"):
            return ("heading", n)
        if n > TURN_MAX:
            return ("heading", n) if n <= 360 else None
        if n >= 100 and bare_turn(text):
            return ("either", (side, n))
        return ("turn", (side, n))
    if intent == "set_autopilot_heading" and bare_turn(text):
        n = _number(slots.get("degrees"))
        side = side_said(text)
        if n is not None and side is not None and 100 <= n <= TURN_MAX:
            return ("either", (side, n))
    return None


_TURN_SLOTS = ("turn", "turn_degrees", "degrees")


def resolve_turn(
    intent: str, slots: Mapping[str, Any], text: str
) -> tuple[str, dict[str, Any]] | None:
    """Turn or heading, by the words, for a model that knows both (the
    owner's choice after v27's first gate, 2026-11-20). The model finds the
    side and the number; which command it is follows the L.1 rules
    (docs/PHASE_PLAN.md), not the model. v27's first candidate read "turn
    right ninety" as heading 090 and "turn left to fifteen" as a turn: the
    heading phrasings it trained on said "turn right to" and "turn left
    heading", and their mirror images were held out, so it took the side for
    the cue instead of "to".

    ``text`` is the normalized text. Returns the intent and slots to use, or
    None to keep the model's:

    - "heading", or "to" right before the number: the heading;
    - otherwise, with one side said: a turn, unless the number can't be one
      (180 or more, or a leading zero: "turn left 024" is heading 024). From
      100 to 179 the plugin asks which (``turn_reading``), as it did;
    - a turn with no side said, or both: the turn without its side, so it is
      refused for the missing side.
    """
    if intent not in ("turn_by", "set_autopilot_heading"):
        return None
    raw = str(slots.get("turn_degrees") or slots.get("degrees") or "").strip()
    n = _number(raw)
    if n is None:
        return None
    rest = {k: v for k, v in slots.items() if k not in _TURN_SLOTS}
    heading = ("set_autopilot_heading", {**rest, "degrees": raw})
    if says_heading(text):
        return heading
    side = side_said(text)
    if side is None:
        return ("turn_by", {**rest, "turn_degrees": raw}) if intent == "turn_by" else None
    if (len(raw) == 3 and raw.startswith("0")) or n > TURN_MAX:
        return heading
    return ("turn_by", {**rest, "turn": side, "turn_degrees": raw})


def climb_reading(intent: str, slots: Mapping[str, Any], text: str) -> tuple[str, int] | None:
    """(direction, feet) for a bare climb or descent whose number could be
    the altitude or the change (up to ``ALTITUDE_CHANGE_MAX``); None
    otherwise. Larger numbers can only be the altitude, as before."""
    if intent not in ("set_autopilot_altitude", "change_altitude_by"):
        return None
    direction = bare_climb(text)
    if direction is None:
        return None
    n = _number(slots.get("altitude") or slots.get("altitude_change"))
    if n is None or not 0 < n <= ALTITUDE_CHANGE_MAX or n % ALTITUDE_STEP:
        return None
    return direction, n


# ---------------------------------------------------------------------------
# Targets, from the aircraft
# ---------------------------------------------------------------------------
def turned(heading: float, side: str, degrees: int) -> int:
    """The heading ``degrees`` to the ``side`` of ``heading``, 1-360."""
    sign = -1 if side == "left" else 1
    return int(round(heading + sign * degrees)) % 360 or 360


def _nearest(value: float, step: int) -> int:
    return int(round(value / step) * step)


def altitude_reference(
    selected: float | None, current: float, vs_fpm: float | None, moving: bool
) -> int:
    """What a change is measured from: the selected altitude while the
    autopilot climbs or descends to it (so "climb another thousand" on the
    way to 11,000 gives 12,000), otherwise the altitude now, to 100 ft."""
    if moving and selected is not None and vs_fpm is not None and abs(vs_fpm) >= MOVING_FPM:
        if (vs_fpm > 0 and selected > current) or (vs_fpm < 0 and selected < current):
            return int(round(selected))
    return _nearest(current, ALTITUDE_STEP)


def changed(reference: float, direction: str, amount: int) -> int:
    return int(round(reference + (amount if direction == "up" else -amount)))


def goes_the_way(direction: str, target: float, current: float) -> bool:
    """A climb ends above us, a descent below."""
    return target > current if direction == "up" else target < current


def valid_altitude(feet: int) -> bool:
    return SLOT_VALIDATORS["altitude"].contains(feet)


def altitude_readings(direction: str, feet: int, reference: int, current: float) -> list[int]:
    """The altitudes a bare climb or descent by ``feet`` can mean, each
    valid and going the way said: the number itself, then the change from
    ``reference``. One left means act on it, two means ask, none means refuse."""
    out = []
    for target in (feet, changed(reference, direction, feet)):
        if valid_altitude(target) and goes_the_way(direction, target, current):
            if target not in out:
                out.append(target)
    return out


def speed_reference(selected: float | None, ias: float, held: bool) -> int:
    """The speed bug when the autothrottle (or the autopilot's speed mode) is
    flying it, otherwise the airspeed now, to 5 kt."""
    if held and selected is not None and selected > 0:
        return int(round(selected))
    return _nearest(ias, SPEED_STEP)
