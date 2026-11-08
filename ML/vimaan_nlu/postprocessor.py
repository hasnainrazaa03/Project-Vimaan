from __future__ import annotations

import re
from typing import Any

from .commands import INSTANCE_WORDS, choice_synonyms, command
from .validators import in_range

# Verbs that imply a choice ("raise" -> gear up, "disconnect" -> autopilot
# off), per intent and slot, from the command table. Per intent because "set"
# means ON only for the parking brake and up/down apply only to gear/flaps.
CHOICE_SYNONYMS: dict[str, dict[str, dict[str, tuple[str, ...]]]] = choice_synonyms()

# Generic verbs valid for every on/off choice.
_GENERIC_STATE: dict[str, tuple[str, ...]] = {
    "on": ("turn on", "switch on"),
    "off": ("turn off", "switch off"),
}


# "com 2", "comm 2", "radio 2" in normalized text.
_PORT_IN_TEXT = re.compile(r"\b(?:com|comm|radio)\s+([1-4])\b")
# "133.1 on 2": "com" lost after "on" (Vosk heard "on com two" as "on on two").
_PORT_AFTER_FREQUENCY = re.compile(r"\b\d{3}\.\d+\s+(?:on\s+)+([1-4])\b")


def port_from_text(slots: dict[str, Any], text: str, intent: str | None) -> dict[str, Any]:
    """Fill an untagged COM port from the text when exactly one is named.

    The model can miss the port in a phrasing it never trained on: v17 left
    it untagged in "frequency 133.1 on com 2" (a held-out template), so the
    plugin's COM 1 default tuned the wrong radio, in flight. "133.1 on 2"
    counts too: in flight Vosk heard "on com two" as "on on two" and COM 1
    was tuned. A tagged port is never overridden, and two different ports
    named leave it to the default.
    """
    cmd = command(intent or "")
    if cmd is None or "com_port" not in cmd.slots or slots.get("com_port"):
        return slots
    lowered = text.lower()
    ports = set(_PORT_IN_TEXT.findall(lowered)) | set(_PORT_AFTER_FREQUENCY.findall(lowered))
    if len(ports) == 1:
        slots["com_port"] = ports.pop()
    return slots


def extract_numbers_from_text(text: str) -> list[str]:
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return numbers


def extract_digit_sequence_frequency(text: str) -> str | None:
    digit_map = {
        "zero": "0",
        "oh": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "niner": "9",
        "nine": "9",
        "point": ".",
        "decimal": ".",
    }

    if "com" in text.lower() or "frequency" in text.lower():
        words = text.lower().split()
        result = []
        found_sequence = False

        for i, word in enumerate(words):
            if word in digit_map:
                result.append(digit_map[word])
                found_sequence = True
            elif found_sequence and len(result) >= 3:
                freq_str = "".join(result)
                if in_range("frequency", freq_str):
                    return freq_str
                result = []
                found_sequence = False

        if len(result) >= 3:
            freq_str = "".join(result)
            if in_range("frequency", freq_str):
                return freq_str

    return None


def _build_matchers(synonyms: dict[str, tuple[str, ...]]) -> list[tuple[re.Pattern[str], str]]:
    """Compiled (regex, canonical) matchers for one choice slot.

    Verbs match as a STEM PREFIX (``\\bengag`` -> engage/engaged/engaging) so
    inflections still count; the short literals on/off/up/down match as WHOLE
    WORDS so ``\\bon\\b`` never fires inside "one". Verbs are tried first,
    longest first (so "disengage" wins over "engage", "shut down" over "shut")."""
    verbs: list[tuple[str, str]] = []
    literals: list[tuple[str, str]] = []
    for canon, syns in synonyms.items():
        literals.append((canon, canon))
        verbs.extend((v, canon) for v in syns)
        verbs.extend((v, canon) for v in _GENERIC_STATE.get(canon, ()))
    verbs.sort(key=lambda c: len(c[0]), reverse=True)
    matchers: list[tuple[re.Pattern[str], str]] = []
    for verb, canon in verbs:
        stem = verb[:-1] if verb.endswith("e") else verb  # drop trailing e for -ing forms
        matchers.append((re.compile(r"\b" + re.escape(stem)), canon))
    for lit, canon in literals:
        matchers.append((re.compile(r"\b" + re.escape(lit) + r"\b"), canon))
    return matchers


_MATCHERS = {
    intent: {slot: _build_matchers(syns) for slot, syns in slots.items()}
    for intent, slots in CHOICE_SYNONYMS.items()
}


def _match_choice(matchers: list[tuple[re.Pattern[str], str]], text: str) -> str | None:
    for rx, canon in matchers:
        if rx.search(text):
            return canon
    return None


def canonicalize_choices(
    slots: dict[str, Any], original_text: str, intent: str | None
) -> dict[str, Any]:
    """Turn each choice slot into its canonical value, filling it from the
    text when the model didn't tag it: the model tags the word spoken
    ("raise", "disconnect"); the plugin needs "up" / "off"."""
    for slot, matchers in _MATCHERS.get(intent or "", {}).items():
        value = slots.get(slot)
        found = _match_choice(matchers, str(value).lower()) if value else None
        if found is None:
            found = _match_choice(matchers, original_text.lower())
        if found is not None:
            slots[slot] = found
    return slots


# Backward-compatible name: fill a missing state/position from the text.
add_implicit_state = canonicalize_choices


def resolve_instance(
    slots: dict[str, Any], original_text: str, intent: str | None
) -> dict[str, Any]:
    """Which engine / autopilot / flight director: the tagged word ("two",
    "second", "both") or, if none was tagged, the first one named in the
    text; defaults to 1 (captain-side / the only system on most aircraft)."""
    cmd = command(intent or "")
    if cmd is None or "instance" not in cmd.slots:
        return slots
    tagged = str(slots.get("instance", "")).lower().split()
    named = [INSTANCE_WORDS[w] for w in tagged if w in INSTANCE_WORDS]
    if not named:
        named = [INSTANCE_WORDS[w] for w in original_text.lower().split() if w in INSTANCE_WORDS]
    slots["instance"] = named[0] if named else "1"
    return slots


def postprocess_slots(
    slots: dict[str, Any], original_text: str, intent: str | None = None
) -> dict[str, Any]:
    numbers = extract_numbers_from_text(original_text)

    for slot_name, slot_value in list(slots.items()):
        if slot_value is None or slot_value == "":
            continue

        slot_str = str(slot_value).strip()

        if " " in slot_str and slot_str.replace(" ", "").isdigit():
            cleaned = slot_str.replace(" ", "")
            slots[slot_name] = cleaned
            continue

        if slot_name == "frequency":
            digit_freq = extract_digit_sequence_frequency(original_text)
            if digit_freq:
                slots[slot_name] = digit_freq
                continue

            freq_numbers = re.findall(r"\d+(?:\.\d+)?", original_text)
            for num in freq_numbers:
                if "." in num and in_range("frequency", num):
                    slots[slot_name] = num
                    break

        if slot_name in ["altitude", "degrees", "flight_level", "com_port"]:
            # The model tagged a valid number that was really said: keep it.
            # Searching for "the first number in range" instead let a stray
            # word win: "eighty fly heading one eight four" -> heading 80
            # (Vosk does insert a leading "eighty", seen in flight).
            if slot_str in numbers and in_range(slot_name, slot_str):
                continue
            if len(numbers) > 0:
                if slot_name == "altitude":
                    for num in numbers:
                        if in_range("altitude", num):
                            slots[slot_name] = num
                            break

                elif slot_name == "degrees":
                    for num in numbers:
                        if in_range("degrees", num):
                            slots[slot_name] = num
                            break

                elif slot_name == "flight_level":
                    for num in reversed(numbers):
                        if in_range("flight_level", num):
                            slots[slot_name] = num
                            break

                elif slot_name == "com_port":
                    for num in numbers:
                        if in_range("com_port", num) and len(num) == 1:
                            slots[slot_name] = num
                            break

    if intent:
        slots = canonicalize_choices(slots, original_text, intent)
        slots = resolve_instance(slots, original_text, intent)
        slots = port_from_text(slots, original_text, intent)

    return slots
