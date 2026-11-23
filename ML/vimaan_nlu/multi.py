"""Two commands in one breath (phase L.3).

Said together, "gear down flaps fifteen" is one utterance to the model, which
reads one command and drops the other: v27 set flaps 15 and never lowered the
gear (typed, 2026-11-22: 19 of 20 such pairs acted on one part). This finds a
second command so the plugin can refuse the utterance ("One command at a
time", L.3.1) instead of acting on half of it.

Words alone can't tell: 14 % of single commands have two command words ("auto
brake", "parking brake", "flight level"), and 9 % have "and" ("go ahead and
arm the speed brakes", "climb and maintain five thousand"). So the utterance is
split where a second command can begin, at "and" or before a command word,
and the model reads each part. It is two commands only if each part is a whole
command by itself:

- the model's confidence clears the plugin's floor;
- an action, not a question;
- one of that command's own words is in the part;
- every slot the command needs is there and valid, and for a command that
  is a choice (on / off, up / down) the choice is said.

And it is one command, not two, when:

- a split falls inside a number ("one thousand eight | hundred feet per
  minute");
- the whole utterance reads as one of the two parts' commands and the other
  part's command words are that command's own ("nav | standby frequency one
  one three point seven" is a NAV standby frequency, not nav mode and a
  frequency; "set vertical speed | climb two thousand" a vertical speed);
- both parts are the same command with the same settings ("set parking |
  brake on").

Two names sharing one verb at an "and" are two commands ("landing and taxi
lights on", "gear and flaps up"): the second part is whole, and the first
names a command but not its on / off or up / down, or names another value of
the second's own setting (another light).

Pure Python: the caller passes its own ``predict(text) -> dict``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .commands import command, required_slots
from .questions import looks_like_question
from .validators import COMMAND_WORDS, has_command_word, validate_slot

# What the plugin says, doing neither (owner's decision 2026-11-22).
REPLY = "One command at a time"
# As the plugin's CONFIDENCE_FLOOR: a part it wouldn't act on isn't a command.
CONFIDENCE_FLOOR = 0.55
JOINERS = frozenset({"and"})
# A part never starts with these: a split there falls inside a number.
NUMBER_WORDS = frozenset(
    "zero oh one two three four five six seven eight nine niner ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty "
    "fifty sixty seventy eighty ninety hundred thousand decimal point".split()
)
# The words of an on / off, up / down choice.
VERB_VALUES = frozenset({"on", "off", "up", "down", "full"})
# Each split is two model runs (parts already read are reused).
MAX_SPLITS = 6

_ALL_COMMAND_WORDS = frozenset().union(*COMMAND_WORDS.values())
_REQUIRED = required_slots()


def _phrasing_words(intent: str) -> frozenset[str]:
    """Every literal word of a command's phrasings (its own words only)."""
    cmd = command(intent)
    if cmd is None:
        return frozenset()
    return frozenset(w for t in cmd.templates for w in _literal(t) if w)


def _literal(template: str) -> list[str]:
    """A phrasing's words with each slot as "" (never part of a pair)."""
    return ["" if "{" in w else w.lower() for w in template.replace("{", " {").split()]


def _pairs_in_phrasings() -> frozenset[tuple[str, str]]:
    """Each two literal words side by side in any command's phrasing:
    "nav standby", "auto brake", "on nav" (in "the ils on nav one ...").
    A split between them cuts one command's phrase."""
    from .commands import COMMANDS

    out: set[tuple[str, str]] = set()
    for cmd in COMMANDS:
        for t in cmd.templates:
            words = _literal(t)
            out.update((a, b) for a, b in zip(words, words[1:], strict=False) if a and b)
    return frozenset(out)


_PHRASE_PAIRS = _pairs_in_phrasings()


def _slots(intent: str) -> dict[str, Any]:
    cmd = command(intent)
    return dict(cmd.slots) if cmd is not None else {}


def _verb_slot(slot: Any) -> bool:
    return slot.kind == "choice" and set(map(str, slot.values)) <= VERB_VALUES


def _named(p: dict[str, Any], text: str) -> bool:
    """An action the model is sure of, with one of its own words said."""
    cmd = command(p["intent"])
    return (
        cmd is not None
        and cmd.actionable
        and p["confidence"] >= CONFIDENCE_FLOOR
        and has_command_word(p["intent"], text)
        and not looks_like_question(text)
    )


def _valid(p: dict[str, Any]) -> bool:
    return all(validate_slot(name, value)[1] for name, value in p["slots"].items())


# Commands whose name says on: "heading mode" is heading mode on.
_ON_BY_NAME = frozenset({"set_autopilot_mode"})
# One kind of command, absolute or relative: L.1 flies a turn as a heading,
# so "turn right thirty degrees ..." read whole as a heading is the turn's.
_KIND = {
    "turn_by": "heading",
    "set_autopilot_heading": "heading",
    "set_compass_heading": "heading",
    "change_altitude_by": "altitude",
    "set_autopilot_altitude": "altitude",
    "set_flight_level": "altitude",
    "change_speed_by": "speed",
    "set_autopilot_speed": "speed",
}


def _kind(intent: str) -> str:
    return _KIND.get(intent, intent)


# Number slots that say which one, not what to do (a number is what to do:
# "flaps fifteen", "vertical speed one thousand").
_WHICH = frozenset({"instance", "nav_port", "com_port"})


def whole(p: dict[str, Any], text: str) -> bool:
    """A command the plugin could act on by itself: every slot it needs, and
    for one that needs none (flaps), something said (up, or fifteen). Here,
    a command with an on / off or up / down must say it, or a number ("flaps
    fifteen"): "alternators ahead" named the generators and nothing to do with
    them. Not one whose name says on ("heading mode")."""
    if not (_named(p, text) and _valid(p)):
        return False
    intent, slots = p["intent"], p["slots"]
    required = _REQUIRED.get(intent, ())
    if any(not slots.get(name) for name in required):
        return False
    defs = _slots(intent)
    verbs = [name for name, slot in defs.items() if _verb_slot(slot)]
    numbers = [n for n, slot in defs.items() if slot.kind == "number" and n not in _WHICH]
    said = any(slots.get(n) for n in verbs) or any(slots.get(n) for n in numbers)
    if verbs and intent not in _ON_BY_NAME and not said:
        return False
    return bool(required) or bool(slots) or not _slots(intent)


def lacks_only_its_choice(p: dict[str, Any], text: str) -> bool:
    """A command named without its on / off or up / down ("gear", "landing
    lights"): what it misses is only that."""
    if not (_named(p, text) and _valid(p)):
        return False
    if not set(text.split()) <= _phrasing_words(p["intent"]):
        return False  # "beacon ahead and ..." was "go ahead and ..."
    slots = _slots(p["intent"])
    verbs = {name for name, slot in slots.items() if _verb_slot(slot)}
    if not verbs or any(p["slots"].get(name) for name in verbs):
        return False
    missing = {name for name in _REQUIRED.get(p["intent"], ()) if not p["slots"].get(name)}
    return missing <= verbs


def another_value(words: list[str], p: dict[str, Any]) -> bool:
    """``words`` name another value of a setting ``p`` has ("beacon" with
    the strobes on): the same command for another light."""
    said = " ".join(w for w in words if w not in {"the", "lights", "light"})
    for name, slot in _slots(p["intent"]).items():
        if slot.kind != "choice" or _verb_slot(slot) or not p["slots"].get(name):
            continue
        for value in map(str, slot.values):
            names = {value, *slot.synonyms.get(value, ())}
            if said in names and value != str(p["slots"][name]):
                return True
    return False


def splits(words: list[str]) -> list[tuple[list[str], list[str], bool]]:
    """Where a second command can begin: around an "and" (first), or
    between any two words; never inside a number. (first, second, at an
    "and"); at most MAX_SPLITS."""
    at_and = [
        (words[:i], words[i + 1 :], True)
        for i, w in enumerate(words)
        if w in JOINERS and 0 < i < len(words) - 1 and words[i + 1] not in NUMBER_WORDS
    ]
    between = [
        (words[:i], words[i:], False)
        for i in range(2, len(words) - 1)
        if words[i] not in NUMBER_WORDS | JOINERS
        and words[i - 1] not in JOINERS
        and (words[i - 1], words[i]) not in _PHRASE_PAIRS
    ]
    out = [
        (a, b, j)
        for a, b, j in at_and + between
        if _ALL_COMMAND_WORDS.intersection(b) and (j or _ALL_COMMAND_WORDS.intersection(a))
    ]
    return out[:MAX_SPLITS]


def _one_command(
    a: dict[str, Any], a_text: str, b: dict[str, Any], b_text: str, w: dict[str, Any], w_text: str
) -> bool:
    """Two whole parts that are one command all the same."""
    if a["intent"] == b["intent"]:
        return a["slots"] == b["slots"]  # the same command twice, the same settings
    if not whole(w, w_text):
        return False
    if set(w["slots"]) - set(_slots(w["intent"])):
        # Read whole, it holds another command's setting: the squawk code
        # with a transponder mode ("squawk seven seven zero zero and
        # transponder alt"). Two commands.
        return False
    if _kind(w["intent"]) not in (_kind(a["intent"]), _kind(b["intent"])):
        return True  # all of it is a third command ("set vertical speed climb two thousand")
    # The other part's own command words (what made it that command, not
    # "set" or "go"): all the whole command's too, and it is that command
    # ("set heading | go ninety degrees" is a heading of 090).
    other, other_text = (b, b_text) if _kind(w["intent"]) == _kind(a["intent"]) else (a, a_text)
    own = COMMAND_WORDS.get(other["intent"], frozenset()).intersection(other_text.split())
    return own <= COMMAND_WORDS.get(w["intent"], frozenset())


def two_commands(
    text: str,
    predict: Callable[[str], dict[str, Any]],
    whole_reading: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """The two commands in ``text``, each {"text", "intent", "slots",
    "confidence"}, or None when it is one command (or nothing).
    ``whole_reading``: the model's reading of all of ``text``, if the caller
    has it (else it is read here)."""
    words = text.lower().split()
    if sum(1 for w in words if w in _ALL_COMMAND_WORDS) < 2 and "and" not in words:
        return None  # one command word: one command at most
    w_text = " ".join(words)
    seen: dict[str, dict[str, Any]] = {}
    if whole_reading is not None:
        seen[w_text] = whole_reading

    def read(part: list[str]) -> tuple[str, dict[str, Any]]:
        s = " ".join(part)
        if s not in seen:
            seen[s] = predict(s)
        return s, seen[s]

    for first, second, at_and in splits(words):
        b_text, b = read(second)
        if not whole(b, b_text):
            continue
        if at_and and another_value(first, b):
            a_text, a = " ".join(first), {"intent": b["intent"], "slots": {}, "confidence": 0.0}
            return _parts((a_text, a), (b_text, b))
        a_text, a = read(first)
        if at_and and lacks_only_its_choice(a, a_text):
            return _parts((a_text, a), (b_text, b))
        if not whole(a, a_text):
            continue
        _, w = read(words)
        if _one_command(a, a_text, b, b_text, w, w_text):
            continue
        return _parts((a_text, a), (b_text, b))
    return None


def _parts(*parts: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"text": t, "intent": p["intent"], "slots": p["slots"], "confidence": p["confidence"]}
        for t, p in parts
    ]
