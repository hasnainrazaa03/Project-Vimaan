"""Per-class validation metrics for the live training stream.

Pure Python (no torch), so they are unit-tested on plain lists. The trainer
logs them once per epoch; the dashboard shows which intents and slots are
still behind while a run trains, not just the averages.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence


def per_intent_accuracy(
    true_ids: Sequence[int], pred_ids: Sequence[int], intent_map: Mapping[str, int]
) -> dict[str, dict[str, float]]:
    """intent -> {"acc", "n"} over the validation rows of that intent."""
    names = {i: name for name, i in intent_map.items()}
    total: Counter[int] = Counter(true_ids)
    right: Counter[int] = Counter(t for t, p in zip(true_ids, pred_ids, strict=True) if t == p)
    return {names.get(i, str(i)): {"acc": right[i] / n, "n": n} for i, n in sorted(total.items())}


def _slot_name(label: str) -> str:
    return label[2:] if label[:2] in ("B-", "I-") else label


def per_slot_f1(
    true_ids: Sequence[int], pred_ids: Sequence[int], slot_map: Mapping[str, int]
) -> dict[str, dict[str, float]]:
    """slot -> {"f1", "n"}, token level, with B- and I- tags of a slot counted
    as that slot. "O" (no slot) is left out: it is most tokens and always
    near 1."""
    names = {i: _slot_name(label) for label, i in slot_map.items()}
    true = [names.get(t, str(t)) for t in true_ids]
    pred = [names.get(p, str(p)) for p in pred_ids]
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for t, p in zip(true, pred, strict=True):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    out = {}
    for slot in sorted((set(true) | set(pred)) - {"O"}):
        denom = 2 * tp[slot] + fp[slot] + fn[slot]
        out[slot] = {"f1": (2 * tp[slot] / denom) if denom else 1.0, "n": tp[slot] + fn[slot]}
    return out
