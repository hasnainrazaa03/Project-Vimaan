"""Tests for the dashboard's live-metrics reader."""

import json

from dashboard import run_metrics


def _write(run_dir, records):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.write("{half-written trailing line")  # tolerated


def test_latest_run_metrics(tmp_path, monkeypatch):
    _write(
        tmp_path / "v9",
        [
            {"type": "meta", "version": "v9", "split": {"train": 1, "val": 1, "test": 1}},
            *[{"type": "step", "step": i, "loss": 1.0 / i, "epoch": 1} for i in range(1, 1001)],
            {
                "type": "epoch",
                "epoch": 1,
                "train_loss": 0.6,
                "val_loss": 0.5,
                "val_intent_acc": 0.9,
                "val_slot_f1": 0.8,
            },
            {"type": "done", "best_val_loss": 0.5},
        ],
    )
    monkeypatch.setattr(run_metrics, "TRAINING_RUNS", tmp_path)
    d = run_metrics.latest_run_metrics()
    assert d["run"] == "v9"
    assert d["meta"]["version"] == "v9"
    assert len(d["steps"]) <= run_metrics.MAX_STEP_POINTS + 1
    assert d["steps"][-1]["step"] == 1000
    assert len(d["epochs"]) == 1 and d["done"]["best_val_loss"] == 0.5


def test_latest_run_metrics_no_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_metrics, "TRAINING_RUNS", tmp_path)
    assert run_metrics.latest_run_metrics() == {"waiting": True}
