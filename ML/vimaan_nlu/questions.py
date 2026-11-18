"""Spot a question, so it is never acted on as a command.

v26 knows no questions (Tier 4 adds them): "is the autopilot off" was read as
"autopilot off" and would have switched it off, "is the parking brake off"
would have released it, and "what's the heading" set heading mode. In the
owner's voice (40 recordings, 2026-11-16) the recogniser kept the question
word in 24 of 28 questions ("is" every time; "are" comes out as the letter
"r"), so a transcript that starts like a question is refused instead.
No command phrasing in the training data starts with one of these words.

What this can't catch: a question whose first word went unheard. Generated
voices lost it almost every time ("is the autopilot off" heard as "the
autopilot off"), and "where" / "do" are not in the grammar yet ("where are
the flaps" heard as "vertical flaps"). Tier 4 adds the words and a key for
questions only.

Pure Python: it works on the transcript, before anything reaches the sim.
"""

from __future__ import annotations

# A transcript starting with one of these is a question. Only "is", "are",
# "what" and "how" are in the Vosk grammar today; the rest come from other
# recognisers (and Tier 4's grammar).
QUESTION_WORDS = frozenset(
    {
        "is",
        "isn't",
        "are",
        "aren't",
        "what",
        "what's",
        "whats",
        "where",
        "where's",
        "how",
        "how's",
        "do",
        "does",
        "did",
        "which",
    }
)
# Vosk hears the owner's "are the ..." as "r the ..." (command grammar) or
# "our the ..." (question grammar). "r" alone is a letter of "r t o" (auto
# brake), so these count only with "the" after them.
_ARE_THE = (("r", "the"), ("our", "the"))
# Hesitations skipped at the start ("uh what's our ground speed" was heard
# "oh what are ground speed").
_FILLER = frozenset({"uh", "um", "er", "oh", "okay", "ok", "so", "hey"})
# And the words the recogniser put in front of commands in real flights
# ("feet engine two off", "eighty engine two off": the generator's
# LEADING_NOISE), so "now is the autopilot on" is still a question.
# Addressing the copilot by name ("vimaan, is the gear down") too.
_LEADING = _FILLER | {"the", "a", "and", "now", "feet", "eighty", "vimaan"}

# A question that isn't one Vimaan answers (or a command said on the question key).
REPLY = "I can't answer that"


def looks_like_question(text: str) -> bool:
    """Whether ``text`` starts like a question ("is the gear down", "what
    are altitude", "r the strobes off"). "Can you ..." is not one: it is how
    pilots phrase commands ("can you turn on the logo lights")."""
    words = text.lower().split()
    while len(words) > 1 and words[0] in _LEADING and tuple(words[:2]) not in _ARE_THE:
        words.pop(0)
    if not words:
        return False
    return words[0] in QUESTION_WORDS or tuple(words[:2]) in _ARE_THE


# A reading named on its own is answered (Tier 4), never taken as a command:
# "what's the heading" came back as "what the heading", and with the question
# word lost as "heading", which v26 read as heading MODE. Switch names
# ("flaps", "gear", "fuel") are left to their commands, which ask which way.
BARE_READINGS: dict[str, tuple[str, dict[str, str]]] = {
    "heading": ("ask_flight_data", {"reading": "heading"}),
    "altitude": ("ask_flight_data", {"reading": "altitude"}),
    "airspeed": ("ask_flight_data", {"reading": "airspeed"}),
    "speed": ("ask_flight_data", {"reading": "airspeed"}),
    "vertical speed": ("ask_flight_data", {"reading": "vertical speed"}),
    "ground speed": ("ask_flight_data", {"reading": "ground speed"}),
    "mach": ("ask_flight_data", {"reading": "mach"}),
    "altimeter": ("ask_flight_data", {"reading": "altimeter"}),
    "wind": ("ask_flight_data", {"reading": "wind"}),
    "temperature": ("ask_flight_data", {"reading": "temperature"}),
    "squawk": ("ask_transponder", {}),
}
# Words around a bare reading that ask, not command ("what is the heading",
# "say altitude", "check squawk").
_AROUND = (
    QUESTION_WORDS
    | _FILLER
    | {
        "the",
        "a",
        "our",
        "r",
        "my",
        "current",
        "say",
        "check",
        "please",
        "now",
        "tell",
        "me",
    }
)


def bare_reading(text: str) -> tuple[str, dict[str, str]] | None:
    """(question intent, slots) when ``text`` names one reading and nothing
    else ("heading", "what are altitude"), else None. A value or a state
    ("heading two seven zero", "altitude hold") is not bare."""
    words = [w for w in text.lower().split() if w not in _AROUND]
    found = BARE_READINGS.get(" ".join(words))
    return None if found is None else (found[0], dict(found[1]))
