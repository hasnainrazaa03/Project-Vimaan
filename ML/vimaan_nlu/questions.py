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
# Vosk hears the owner's "are the ..." as "r the ...": "r" alone is a letter
# of "r t o" (auto brake), so it counts only with "the" after it.
_LETTER_ARE = ("r", "the")
# Hesitations skipped at the start ("uh what's our ground speed" was heard
# "oh what are ground speed").
_FILLER = frozenset({"uh", "um", "er", "oh", "okay", "ok", "so", "hey"})

REPLY = "I can't answer questions yet"


def looks_like_question(text: str) -> bool:
    """Whether ``text`` starts like a question ("is the gear down", "what
    are altitude", "r the strobes off"). "Can you ..." is not one: it is how
    pilots phrase commands ("can you turn on the logo lights")."""
    words = text.lower().split()
    while words and words[0] in _FILLER:
        words.pop(0)
    if not words:
        return False
    return words[0] in QUESTION_WORDS or tuple(words[:2]) == _LETTER_ARE
