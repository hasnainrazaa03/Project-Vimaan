"""Read the trainer's live metrics stream for the dashboard.

train_nlu_model.py streams one JSON object per line to
``ML/training_runs/<version>/metrics.jsonl`` (meta / step / epoch / done). The
dashboard launches that exact script, so every dashboard-run also produces this
feed. We surface the newest run's stream for the live training charts.
"""

from __future__ import annotations

import json
from typing import Any

from .paths import TRAINING_RUNS

MAX_STEP_POINTS = 800  # downsample the step curve so the payload stays small


def _newest_run():
    if not TRAINING_RUNS.is_dir():
        return None
    runs = [p for p in TRAINING_RUNS.iterdir() if (p / "metrics.jsonl").is_file()]
    if not runs:
        return None
    return max(runs, key=lambda p: (p / "metrics.jsonl").stat().st_mtime)


def latest_run_metrics() -> dict[str, Any]:
    run = _newest_run()
    if run is None:
        return {"waiting": True}

    meta = done = None
    steps: list[dict] = []
    epochs: list[dict] = []
    try:
        with open(run / "metrics.jsonl", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # half-written trailing line while the trainer flushes
                t = rec.get("type")
                if t == "meta":
                    meta = rec
                elif t == "step":
                    steps.append(rec)
                elif t == "epoch":
                    epochs.append(rec)
                elif t == "done":
                    done = rec
    except FileNotFoundError:
        return {"waiting": True}

    if len(steps) > MAX_STEP_POINTS:
        k = len(steps) / MAX_STEP_POINTS
        idx = sorted({int(i * k) for i in range(MAX_STEP_POINTS)} | {len(steps) - 1})
        steps = [steps[i] for i in idx]

    return {
        "run": run.name,
        "meta": meta,
        "steps": [{"step": s["step"], "loss": s["loss"], "epoch": s["epoch"]} for s in steps],
        "epochs": epochs,
        "done": done,
    }
