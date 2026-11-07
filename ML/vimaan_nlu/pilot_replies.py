"""Short replies that answer Vimaan rather than command the aircraft.

"Override" executes a command a safety rule is holding ("Confirm gear up, low
altitude"); "cancel" drops a held command or an open "to or two?" question.
They are matched on the whole transcript, never sent to the model: a reply
names no system, so a classifier has nothing to go on, and a fixed list
can't drift with retraining. A longer utterance ("confirm gear up") is not a
reply and goes to the model as a command.

What can't be overridden: validator limits (a heading of 400) and commands
the aircraft can't perform are impossible, not risky, so there is nothing to
confirm.
"""

from __future__ import annotations

OVERRIDE = frozenset(
    {
        "override",
        "override confirm",
        "confirm override",
        "confirm",
        "confirmed",
        "affirm",
        "affirmative",
    }
)
CANCEL = frozenset({"cancel", "negative", "disregard", "abort"})
# Words that may come with a reply without changing it.
_FILLER = {"please", "uh", "um", "er"}

# For the recogniser's grammar (vimaan_nlu.stt_vocabulary): distinctive words
# only. Fillers stay out of the grammar, or Vosk could start inserting "uh"
# into ordinary commands that no model has been trained on.
REPLY_WORDS = frozenset(w for phrase in OVERRIDE | CANCEL for w in phrase.split())


def pilot_reply(text: str) -> str | None:
    """Return "override", "cancel", or None when ``text`` is anything else."""
    words = [w for w in text.lower().split() if w not in _FILLER]
    phrase = " ".join(words)
    if phrase in OVERRIDE:
        return "override"
    if phrase in CANCEL:
        return "cancel"
    return None
