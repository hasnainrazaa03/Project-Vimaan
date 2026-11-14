"""How a v2 dataset (ML/data/generate_v2.py) is split for training.

Rows say which split they belong to: "test" (held-out templates, the gate's
new phrasings), "dev" (more held-out templates, from data v13) or "train".
With dev rows, validation is the dev rows. The checkpoint kept is then the one
best on phrasings it never saw, which is what the gate grades. Validation on
rows of trained templates sat at ~99.99% every epoch, so which epoch was kept
said nothing about new phrasings.

Pure Python, so it can be tested without torch.
"""

from __future__ import annotations

from typing import Any

Rows = list[dict[str, Any]]


def template_split(rows: Rows) -> tuple[Rows, Rows, Rows] | None:
    """(train, validation, test) when the dataset has dev rows, else None
    (older datasets: the caller takes a random validation slice)."""
    dev = [r for r in rows if r.get("split") == "dev"]
    if not dev:
        return None
    test = [r for r in rows if r.get("split") == "test"]
    train = [r for r in rows if r.get("split") not in ("test", "dev")]
    return train, dev, test
