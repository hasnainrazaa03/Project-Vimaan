"""The words the offline recogniser (Vosk) is allowed to output.

Vosk can decode against a *grammar*: a list of words, instead of its full
~100k-word vocabulary. With only cockpit words available, mishearings like
"slaps" (flaps) or "Secom" (set com) cannot happen, and measured on synthetic
voices end-to-end intent+slot accuracy went from 62/108 (free) to 94/108.
``[unk]`` stays in the grammar so that speech which isn't a command decodes to
``[unk]`` rather than being forced into command words (without it, "let me
check the chart" came out as "retract start" = start engine 1).

The word list is generated from the training schema
(``ML/config/schema_config.py``) by ``build_vocabulary`` and committed as
``command_words.txt`` next to this module, because the plugin ships
``vimaan_nlu`` but not the schema. ``tests/test_stt_vocabulary.py`` fails when
the two drift apart; regenerate with ``python scripts/build_stt_vocabulary.py``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

VOCABULARY_FILE = os.path.join(os.path.dirname(__file__), "command_words.txt")
UNKNOWN = "[unk]"

# Numbers as pilots say them; the normalizer turns these back into digits.
NUMBER_WORDS = (
    "zero one two three four five six seven eight nine niner ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty "
    "forty fifty sixty seventy eighty ninety hundred thousand decimal point"
).split()
# Words the normalizer / postprocessor understand that templates don't contain.
EXTRA_WORDS = "to the and a first second left right both com radio feet".split()
# Words left out of the grammar because the recogniser hears command words
# as them. In-sim: "two" came back as "do" every time ("autopilot do on"), and
# "flight" as "like" ("like director one on"). Both only appear in
# conversational templates ("do you have the time", "what's the weather like").
EXCLUDED_WORDS = {"do", "like"}
_DIGIT_WORDS = dict(enumerate("zero one two three four five six seven eight nine".split()))


def _words(text: str) -> list[str]:
    text = re.sub(r"\{\w+\}", " ", text.lower()).replace("-", " ")
    out = []
    for w in text.split():
        if w.isdigit():
            out += [_DIGIT_WORDS[int(d)] for d in w]
        elif w.isalpha():
            out.append(w)
    return out


def build_vocabulary(schema: Mapping[str, Any], phrases: Iterable[str] = ()) -> list[str]:
    """Every word in the schema's templates, categorical slot values and
    synonyms, plus ``phrases`` (generator prefixes/suffixes), number words and
    connective words. Sorted, de-duplicated, lowercase."""
    words: set[str] = set(NUMBER_WORDS) | set(EXTRA_WORDS)
    for intent in schema.values():
        for template in intent.get("templates", []):
            words.update(_words(template))
        for slot in intent.get("slots", {}).values():
            if slot.get("type") == "categorical":
                for value in slot.get("values", []):
                    words.update(_words(value))
            for synonyms in slot.get("synonyms", {}).values():
                for synonym in synonyms:
                    words.update(_words(synonym))
    for phrase in phrases:
        words.update(_words(phrase))
    return sorted(words - EXCLUDED_WORDS)


def load_vocabulary(path: str = VOCABULARY_FILE) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [w for w in (line.strip() for line in fh) if w and not w.startswith("#")]


def grammar(words: Iterable[str] | None = None) -> list[str]:
    """The Vosk grammar: the command words plus ``[unk]``."""
    return [*(load_vocabulary() if words is None else words), UNKNOWN]
