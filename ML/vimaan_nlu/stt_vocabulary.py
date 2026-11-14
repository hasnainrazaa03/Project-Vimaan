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

import hashlib
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

VOCABULARY_FILE = os.path.join(os.path.dirname(__file__), "command_words.txt")
UNKNOWN = "[unk]"

# Numbers as pilots say them; the normalizer turns these back into digits.
# "oh" is a spoken zero ("course one oh nine"): the training data says it, so
# without it here "one oh nine" could only be heard as "one nine" (course 19).
NUMBER_WORDS = (
    "zero oh one two three four five six seven eight nine niner ten eleven twelve "
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
# Terms the grammar holds whole, as the pilot says them. Vosk turns the
# grammar's lines into a small bigram model, so a line "anti ice" makes that
# pair likely, while a word on a line of its own is as likely anywhere. The
# words of a term can still be heard alone (by backoff), but as loose words
# "ice" and "fuel" turned up unsaid: a garbled first word in the owner's
# recordings came back "ice director on" and "fuel com ...". A word that is
# also said on its own somewhere keeps its own line too.
JOINED_TERMS = ("anti ice", "fuel pump", "fuel pumps", "fuel boost")
_DIGIT_WORDS = dict(enumerate("zero one two three four five six seven eight nine".split()))


def _words(text: str) -> list[str]:
    text = re.sub(r"\{[^}]*\}", " ", text.lower()).replace("-", " ")
    out = []
    for w in text.split():
        if w.isdigit():
            out += [_DIGIT_WORDS[int(d)] for d in w]
        elif w.isalpha():
            out.append(w)
    return out


def _loose_and_joined(
    texts: Iterable[list[str]], terms: Iterable[str]
) -> tuple[set[str], set[str]]:
    """Words said outside every term, and the terms that occur at all."""
    term_words = [t.split() for t in terms]
    loose: set[str] = set()
    found: set[str] = set()
    for words in texts:
        covered = [False] * len(words)
        for term in term_words:
            n = len(term)
            for i in range(len(words) - n + 1):
                if words[i : i + n] == term:
                    covered[i : i + n] = [True] * n
                    found.add(" ".join(term))
        loose.update(w for w, c in zip(words, covered, strict=True) if not c)
    return loose, found


def build_vocabulary(
    schema: Mapping[str, Any],
    phrases: Iterable[str] = (),
    joined: Iterable[str] = JOINED_TERMS,
) -> list[str]:
    """Every word in the schema's templates, categorical slot values and
    synonyms, plus ``phrases`` (generator prefixes/suffixes), number words and
    connective words; a word said only inside a ``joined`` term is entered as
    that term instead. Sorted, de-duplicated, lowercase."""
    from vimaan_nlu.pilot_replies import REPLY_WORDS

    texts: list[list[str]] = [[w] for w in (*NUMBER_WORDS, *EXTRA_WORDS, *REPLY_WORDS)]
    for intent in schema.values():
        texts += [_words(t) for t in intent.get("templates", [])]
        for slot in intent.get("slots", {}).values():
            if slot.get("type") == "categorical":
                texts += [_words(v) for v in slot.get("values", [])]
            for synonyms in slot.get("synonyms", {}).values():
                texts += [_words(s) for s in synonyms]
    texts += [_words(p) for p in phrases]
    loose, terms = _loose_and_joined(texts, joined)
    return sorted((loose | terms) - EXCLUDED_WORDS)


def entry_words(entries: Iterable[str]) -> set[str]:
    """The single words of grammar entries ("anti ice" -> anti, ice)."""
    return {w for e in entries for w in e.split()}


def load_vocabulary(path: str = VOCABULARY_FILE) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [w for w in (line.strip() for line in fh) if w and not w.startswith("#")]


def grammar(words: Iterable[str] | None = None) -> list[str]:
    """The Vosk grammar: the command words plus ``[unk]``."""
    return [*(load_vocabulary() if words is None else words), UNKNOWN]


def grammar_sha256(words: Iterable[str] | None = None) -> str:
    """Fingerprint of the grammar. The real-voice set records the one it was
    heard with, so a grammar change can't leave it grading old transcripts."""
    return hashlib.sha256("\n".join(grammar(words)).encode()).hexdigest()
