"""Train until dev stops improving, then settle: the "wsd" schedule.

Warmup, steady, decay. The rate warms up, then stays at its peak while dev is
checked every so often. When dev hasn't improved for ``patience`` checks, the
model has converged at that rate: the rate then falls linearly to zero over
``decay_steps`` so the weights settle, and training stops. A run that keeps
improving goes on until the cap (``total_steps``), decaying over its last
``decay_steps``.

Why: with "linear" the rate falls over a fixed number of epochs chosen in
advance. The best dev score came at epoch 1-5 of 10 in every run up to v26, so
the checkpoint kept was a snapshot taken at 60-90% of the peak rate, while the
weights were still moving. Which held-out phrasings it got right was partly
luck: v26-g lost "fly at ... thousand" that v26-h and v26-i, the same recipe,
got right. Here the decay comes when dev says the run has converged.

Pure Python, so it can be tested without torch. Scores are higher-is-better
(pass the negative of a loss).
"""

from __future__ import annotations


class Convergence:
    def __init__(
        self,
        *,
        patience: int,
        total_steps: int,
        warmup_steps: int,
        decay_steps: int,
        min_gain: float = 0.0,
    ) -> None:
        if decay_steps < 1 or decay_steps > total_steps:
            raise ValueError("decay_steps must be between 1 and total_steps")
        self.patience = patience
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_gain = min_gain
        self.best: float | None = None  # the best score seen (what is saved)
        self._bar: float | None = None  # the score a check must beat by min_gain
        self.checks_without_gain = 0
        self.decay_start: int | None = None
        self.reason: str | None = None  # "converged" or "cap", once decaying

    # ------------------------------------------------------------------
    def factor(self, step: int) -> float:
        """The fraction of the peak rate used at ``step`` (0-based)."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if self.decay_start is None or step < self.decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - self.decay_start) / self.decay_steps)

    @property
    def decaying(self) -> bool:
        return self.decay_start is not None

    def check(self, step: int, score: float) -> bool:
        """Record dev's score after ``step`` steps. Returns True when it is
        the best so far (save the weights). Starts the decay once dev has gone
        ``patience`` checks without beating its best by more than ``min_gain``.
        """
        is_best = self.best is None or score > self.best
        if is_best:
            self.best = score
        if self._bar is None or score > self._bar + self.min_gain:
            self._bar = score
            self.checks_without_gain = 0
        else:
            self.checks_without_gain += 1
        if not self.decaying and self.checks_without_gain >= self.patience:
            self._start_decay(step, "converged")
        return is_best

    def step_done(self, step: int) -> None:
        """Call after each training step: starts the decay at the cap."""
        if not self.decaying and step >= self.total_steps - self.decay_steps:
            self._start_decay(step, "cap")

    def finished(self, step: int) -> bool:
        return self.decaying and step >= self.decay_start + self.decay_steps

    def _start_decay(self, step: int, reason: str) -> None:
        self.decay_start = step
        self.reason = reason
