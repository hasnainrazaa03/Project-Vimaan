"""Tests for the live training monitor's metrics parser (pure stdlib)."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ML"))

import training_monitor as tm  # noqa: E402


def _write(run_dir, records, trailing_garbage=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        if trailing_garbage:
            f.write("{half-written line while trainer flushes")  # no newline


def test_read_metrics_parses_and_downsamples(tmp_path):
    run = tmp_path / "v1"
    recs = [{"type": "meta", "version": "v1", "split": {"train": 1, "val": 1, "test": 1}}]
    recs += [{"type": "step", "epoch": 1, "step": i, "loss": 1.0 / i} for i in range(1, 2001)]
    recs += [
        {
            "type": "epoch",
            "epoch": 1,
            "train_loss": 0.5,
            "val_loss": 0.6,
            "val_intent_acc": 0.9,
            "val_slot_f1": 0.8,
            "best": True,
        },
        {"type": "done", "best_val_loss": 0.6},
    ]
    _write(run, recs, trailing_garbage=True)

    d = tm._read_metrics(str(run))
    assert d["meta"]["version"] == "v1"
    assert len(d["steps"]) <= tm.MAX_STEP_POINTS + 1  # downsampled from 2000
    assert d["steps"][-1]["step"] == 2000  # always keeps the latest point
    assert len(d["epochs"]) == 1
    assert d["done"]["best_val_loss"] == 0.6  # garbage trailing line tolerated


def test_read_metrics_missing_file_is_safe(tmp_path):
    (tmp_path / "v1").mkdir()
    d = tm._read_metrics(str(tmp_path / "v1"))
    assert d["meta"] is None and d["steps"] == [] and d["epochs"] == []


def test_newest_run_picks_dir_with_metrics(tmp_path):
    (tmp_path / "v1").mkdir()  # no metrics file -> ignored
    _write(tmp_path / "v2", [{"type": "meta"}])
    assert tm._newest_run(str(tmp_path)).endswith("v2")
    assert tm._newest_run(str(tmp_path / "does-not-exist")) is None
