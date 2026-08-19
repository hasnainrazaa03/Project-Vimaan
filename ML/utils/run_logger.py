"""Append-only JSONL metrics stream for a training run.

The trainer writes one JSON object per line (meta / step / epoch / done) and
flushes immediately so the live monitor (``ML/training_monitor.py``) can tail it
in real time. Logging is best-effort: a failure here must never crash training.
"""

from __future__ import annotations

import json
import os


class RunLogger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, "metrics.jsonl")
        self._f = None
        try:
            os.makedirs(run_dir, exist_ok=True)
            # Truncate any stale file from a previous run of the same version.
            self._f = open(self.path, "w", encoding="utf-8")
        except Exception as e:  # pragma: no cover - disk edge cases
            print(f"[run_logger] disabled ({e})")

    def log(self, **record):
        if self._f is None:
            return
        try:
            self._f.write(json.dumps(record) + "\n")
            self._f.flush()
        except Exception:
            pass

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None
