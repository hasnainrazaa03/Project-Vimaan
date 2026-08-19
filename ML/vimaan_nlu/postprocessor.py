import re

from .validators import in_range

# Per-intent action-verb / literal -> canonical `state`, mirrored from
# config/schema_config.py (kept inline so the runtime/plugin stays self-contained
# — must stay in sync with the schema synonyms). Intent-awareness is the fix for
# the biggest slot-recall hole: the generator writes verbs like "engage",
# "deploy", "ignite", "release" into the text but stores the canonical label
# ("on"/"down"/...), which the aligner never tagged. It is also why the map must
# be per-intent: "set" implies ON only for the parking brake, and up/down apply
# only to gear/flaps — a global map would mislabel "set flaps up".
STATE_SYNONYMS = {
    "toggle_landing_gear": {
        "up": ["raise", "retract", "stow"],
        "down": ["lower", "extend", "deploy", "drop"],
    },
    "toggle_flaps": {"up": ["retract", "raise"], "down": ["extend", "lower", "deploy"]},
    "toggle_autopilot_1": {"on": ["engage", "activate"], "off": ["disengage", "deactivate"]},
    "toggle_autopilot_2": {"on": ["engage", "activate"], "off": ["disengage", "deactivate"]},
    "toggle_flight_director_1": {"on": ["enable", "activate"], "off": ["disable", "deactivate"]},
    "toggle_flight_director_2": {"on": ["enable", "activate"], "off": ["disable", "deactivate"]},
    "toggle_parking_brake": {"on": ["engage", "set"], "off": ["release", "disengage"]},
    "toggle_engine_1": {
        "on": ["start", "ignite"],
        "off": ["stop", "shut down", "shut", "cut", "kill"],
    },
    "toggle_engine_2": {
        "on": ["start", "ignite"],
        "off": ["stop", "shut down", "shut", "cut", "kill"],
    },
}

# Generic verbs valid across every toggle intent.
_GENERIC_STATE = {"on": ["turn on", "switch on"], "off": ["turn off", "switch off"]}

# Intents whose `state` slot may be implied by an action verb ("raise the gear"
# -> up). Derived from STATE_SYNONYMS so it can never drift out of sync.
IMPLICIT_STATE_INTENTS = frozenset(STATE_SYNONYMS)

# Flat verb -> state map (backward-compatible export; derived, so it stays in
# sync). Safe to flatten because no verb maps to two different states.
ACTION_STATE_MAP = {
    verb: canon
    for mapping in STATE_SYNONYMS.values()
    for canon, verbs in mapping.items()
    for verb in verbs
}
ACTION_STATE_MAP.update({v: "on" for v in _GENERIC_STATE["on"]})
ACTION_STATE_MAP.update({v: "off" for v in _GENERIC_STATE["off"]})


def extract_numbers_from_text(text):
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return numbers


def extract_digit_sequence_frequency(text):
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


# Numbered toggle intents whose instance (1/2) is stated explicitly in the
# utterance. The model occasionally confuses the variants (engine 1<->2,
# FD 1<->2, AP 1<->2); the spoken number is authoritative, so we trust it.
_NUMBERED_INTENT_RE = re.compile(r"^(toggle_(?:autopilot|flight_director|engine))_(\d+)$")


def correct_numbered_intent(intent, normalized_text):
    """Fix a mis-numbered toggle intent from the instance number in the text.

    For ``toggle_engine_N`` / ``toggle_autopilot_N`` / ``toggle_flight_director_N``,
    if the normalized utterance names a different instance (1 or 2), return the
    matching variant. No-op for other intents or when no instance digit is
    present, so it is safe to apply unconditionally.
    """
    if not intent:
        return intent
    match = _NUMBERED_INTENT_RE.match(intent)
    if not match:
        return intent
    base, predicted = match.group(1), match.group(2)
    spoken = re.findall(r"\b([12])\b", normalized_text or "")
    if spoken and spoken[0] != predicted:
        return f"{base}_{spoken[0]}"
    return intent


def _build_state_matchers(intent):
    """Compiled (regex, canonical) matchers for a toggle intent.

    Verbs match as a STEM PREFIX (``\\bengag`` -> engage/engaged/engaging/engages,
    ``\\bdeploy`` -> deploy/deployed/deploying) because the paraphraser inflects
    them heavily; a whole-word match would miss "engaged"/"deploying". The short
    literals on/off/up/down instead match as WHOLE WORDS so ``\\bon\\b`` never
    fires inside "one". Verbs are tried first, longest first (so "disengage"
    wins over "engage" and "shut down" over "shut")."""
    verbs, literals = [], []
    for canon, syns in STATE_SYNONYMS[intent].items():
        literals.append((canon, canon))
        verbs.extend((v, canon) for v in syns)
        verbs.extend((v, canon) for v in _GENERIC_STATE.get(canon, []))
    verbs.sort(key=lambda c: len(c[0]), reverse=True)

    matchers = []
    for verb, canon in verbs:
        stem = verb[:-1] if verb.endswith("e") else verb  # drop trailing e for -ing forms
        matchers.append((re.compile(r"\b" + re.escape(stem)), canon))
    for lit, canon in literals:
        matchers.append((re.compile(r"\b" + re.escape(lit) + r"\b"), canon))
    return matchers


_STATE_MATCHERS = {intent: _build_state_matchers(intent) for intent in STATE_SYNONYMS}


def add_implicit_state(slots, original_text, intent):
    """Fill a missing ``state`` slot from an action verb / literal in the text.

    Only for toggle intents, only when the model didn't already extract a state.
    Intent-aware: ``set`` implies ON only for the parking brake and ``down``
    never applies to an autopilot.
    """
    matchers = _STATE_MATCHERS.get(intent)
    if matchers is None:
        return slots
    if slots.get("state"):
        return slots

    text_lower = original_text.lower()
    for rx, canon in matchers:
        if rx.search(text_lower):
            slots["state"] = canon
            return slots
    return slots


def postprocess_slots(slots, original_text, intent=None):
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
        slots = add_implicit_state(slots, original_text, intent)

    return slots
