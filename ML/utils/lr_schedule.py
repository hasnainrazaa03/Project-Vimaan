"""Learning-rate schedules for the trainer.

"constant" keeps the rate fixed for the whole run (the recipe up to v21).
"linear" warms up over the first steps and then falls linearly to zero at the
last step, as BERT-family models are usually fine-tuned. A fixed rate leaves
the model moving to the end. Validation then swings from epoch to epoch (v20:
0.0005 -> 0.012 -> 0.001), and which held-out phrasings a run gets right comes
partly down to which epoch scored best.

Pure Python, so it can be tested without torch.
"""

from __future__ import annotations

SCHEDULES = ("constant", "linear")


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """The fraction of the peak rate used at ``step`` (0-based) under the linear
    schedule: it rises to 1 over ``warmup_steps``, then falls to 1/(the number
    of decay steps) at the last step, and never goes below 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
