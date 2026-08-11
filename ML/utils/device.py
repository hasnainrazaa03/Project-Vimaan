"""Torch device selection with Apple-Silicon (MPS) support.

The trainer historically selected only ``cuda`` else ``cpu``, so on a Mac it
ran on CPU and left the Apple GPU idle. This picks the best available backend:
CUDA (NVIDIA) > MPS (Apple Metal GPU) > CPU, and enables MPS's CPU fallback so
any op the Metal backend hasn't implemented yet transparently runs on CPU
instead of raising.
"""

from __future__ import annotations

import os


def pick_device(prefer_mps: bool = True):
    """Return the best available ``torch.device``: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if prefer_mps and mps is not None and mps.is_available():
        # Let unsupported ops fall back to CPU rather than erroring mid-run.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(name: str | None):
    """Resolve an explicit device name ('cpu'/'mps'/'cuda'), or auto-pick when
    ``name`` is falsy or 'auto'. Lets the trainer expose a ``--device`` escape
    hatch (e.g. force ``cpu`` if MPS misbehaves on a long run)."""
    import torch

    if name and str(name).lower() != "auto":
        return torch.device(str(name).lower())
    return pick_device()
