"""Averaging the weights of several trained models into one ("model soup").

Runs of one recipe each miss a different couple of held-out phrasings
(v22-v24: 2-3 of 87 templates below 99%, never the same one twice), so which
run passes the gate is luck. Averaging their weights keeps what they agree on
and evens out what each got wrong by chance; the result is one model of the
same size and speed. It only works for models fine-tuned from the same base
with the same labels, which ``check_compatible`` enforces.

Pure Python, so it can be tested without torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Files that must be byte-identical in every member: same labels in the same
# order, same architecture, same tokenizer.
SHARED_FILES = (
    "intent_map.json",
    "slot_map.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def check_compatible(dirs: Sequence[Path]) -> None:
    """Raise ValueError unless there are 2+ members and they share SHARED_FILES."""
    if len(dirs) < 2:
        raise ValueError("a soup needs at least two models")
    first = dirs[0]
    for name in SHARED_FILES:
        want = (first / name).read_bytes()
        for d in dirs[1:]:
            if (d / name).read_bytes() != want:
                raise ValueError(f"{d.name}/{name} differs from {first.name}/{name}")


def average(state_dicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Key-by-key mean of the members' weights (anything that adds and divides)."""
    keys = set(state_dicts[0])
    for sd in state_dicts[1:]:
        if set(sd) != keys:
            raise ValueError("the models do not have the same weights")
    return {k: sum(sd[k] for sd in state_dicts) / len(state_dicts) for k in state_dicts[0]}
