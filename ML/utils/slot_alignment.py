"""Char-span slot-label alignment for training (offset-mapping based).

Replaces the old approach in ``train_nlu_model.py`` that built a word->slot map
from ``text.split()`` indices but applied it against the tokenizer's
``word_ids()`` (a different segmentation), and anchored slot values with a raw
``str.find`` that matched substrings inside other tokens — e.g. the ``on`` in
``one`` got labeled ``B-state``, and the ``1`` inside ``128.50`` got labeled
``com_port``. Both corrupted exactly the tokens that carry the signal.

This module is pure-Python (no torch/transformers) so it is unit-testable. The
training dataset calls the tokenizer with ``return_offsets_mapping=True`` and
feeds the resulting ``(start, end)`` char offsets here.
"""

from __future__ import annotations

import re


def find_slot_char_spans(text: str, slots: dict | None):
    """Return ``[(start, end, slot_name)]`` char spans of each slot value in
    ``text``, matched at WORD BOUNDARIES.

    Word boundaries are what fix the substring bug: ``\\bon\\b`` will not match
    inside ``one``, and ``\\b1\\b`` will not match inside ``128``. The first
    boundary-aligned occurrence of each value is used.
    """
    text_lower = text.lower()
    spans = []
    for name, value in (slots or {}).items():
        value_str = str(value).lower().strip()
        if not value_str:
            continue
        match = re.search(r"\b" + re.escape(value_str) + r"\b", text_lower)
        if match:
            spans.append((match.start(), match.end(), name))
    return spans


def align_offsets_to_labels(offsets, spans, slot_map):
    """Map tokenizer ``(start, end)`` offsets to BIO label ids via char spans.

    - Special/padding tokens (``start == end``) get ``-100`` (ignored in loss).
    - A token whose char span sits fully inside a slot span gets ``B-``/``I-``
      of that slot (``B-`` on the slot's first token, ``I-`` after).
    - Everything else gets ``O``.
    """
    o_id = slot_map["O"]
    labels = []
    prev_span = None
    for start, end in offsets:
        if start == end:  # [CLS]/[SEP]/[PAD]
            labels.append(-100)
            prev_span = None
            continue
        matched = None
        for span in spans:
            if span[0] <= start and end <= span[1]:
                matched = span
                break
        if matched is None:
            labels.append(o_id)
            prev_span = None
        else:
            name = matched[2]
            if matched is prev_span:
                labels.append(slot_map[f"I-{name}"])
            else:
                labels.append(slot_map[f"B-{name}"])
                prev_span = matched
    return labels
