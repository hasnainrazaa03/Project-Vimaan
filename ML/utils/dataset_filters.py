"""Dataset-generation quality filters (PR-D / audit T1.3).

For numbered toggle intents the instance number (1/2) lives in the TEMPLATE
text, not a slot — so free paraphrase/augmentation can silently drop or flip it,
producing e.g. text="start the engine" labeled ``toggle_engine_1`` AND another
labeled ``toggle_engine_2``. The model then has no signal for the instance and
guesses (the engine/AP/FD 1↔2 confusion). This filter rejects a numbered-toggle
row whose text no longer contains its instance number.

Pure-Python + unit-testable.
"""

import re

_NUMBERED_RE = re.compile(r"^toggle_(?:autopilot|flight_director|engine)_(\d+)$")
_WORD_FOR_DIGIT = {"1": "one", "2": "two", "3": "three", "4": "four"}


def instance_number_ok(intent, text):
    """True if the row may be kept.

    Returns True for any non-numbered intent, or when the numbered intent's
    instance (e.g. the ``1`` in ``toggle_engine_1``) appears in ``text`` as a
    digit or its spelled word. Returns False when a numbered toggle has lost its
    instance and should be dropped from the training set.
    """
    match = _NUMBERED_RE.match(intent or "")
    if not match:
        return True
    n = match.group(1)
    text_lower = (text or "").lower()
    if re.search(r"\b" + re.escape(n) + r"\b", text_lower):
        return True
    word = _WORD_FOR_DIGIT.get(n)
    return bool(word and re.search(r"\b" + word + r"\b", text_lower))
