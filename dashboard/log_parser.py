"""Parse stdout lines from ML/train_nlu_model.py into structured events."""

from __future__ import annotations

import re

_TRAIN_LOSS = re.compile(r"^Epoch (\d+) - Average Training Loss: ([0-9.]+)")
_VAL_LOSS = re.compile(r"^Epoch (\d+) - Average Validation Loss: ([0-9.]+)")
_SAVED = re.compile(r"^Validation loss improved! Saving best model to (.+)$")
_NO_IMPROVE = re.compile(r"^Validation loss did not improve\. Count: (\d+)/(\d+)")
_EARLY_STOP = re.compile(r"^Early stopping triggered after (\d+) epochs\.")


def parse_line(line: str) -> dict | None:
    """Map a stdout line to an event dict, or None if it doesn't match."""
    s = line.strip()
    if not s:
        return None
    if m := _TRAIN_LOSS.match(s):
        return {"type": "train_loss", "epoch": int(m.group(1)), "value": float(m.group(2))}
    if m := _VAL_LOSS.match(s):
        return {"type": "val_loss", "epoch": int(m.group(1)), "value": float(m.group(2))}
    if m := _SAVED.match(s):
        return {"type": "checkpoint_saved", "path": m.group(1)}
    if m := _NO_IMPROVE.match(s):
        return {"type": "no_improve", "count": int(m.group(1)), "patience": int(m.group(2))}
    if m := _EARLY_STOP.match(s):
        return {"type": "early_stop", "epoch": int(m.group(1))}
    return None
