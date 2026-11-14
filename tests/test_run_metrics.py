"""The dashboard's live-metrics reader (dashboard/run_metrics.py) and the
evals catalogue (dashboard/evals.py), on synthetic runs and models."""

import json
import os
import time

import pytest

from dashboard import evals, paths, run_metrics


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "TRAINING_RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "MODELS_ROOT", tmp_path / "models")
    (tmp_path / "runs").mkdir()
    (tmp_path / "models").mkdir()
    run_metrics._cache.clear()
    return tmp_path


def write(path, records, trailing=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.write(trailing)


META = {
    "type": "meta",
    "version": "v9",
    "device": "mps",
    "epochs": 3,
    "steps_per_epoch": 100,
    "patience": 2,
    "batch_size": 16,
    "split": {"train": 1, "val": 1, "test": 1},
}


def steps(start, end, epoch, t0=0.0):
    return [
        {"type": "step", "step": i, "epoch": epoch, "loss": 1.0 / i, "elapsed": t0 + i * 0.1}
        for i in range(start, end + 1)
    ]


def epoch(n, val_loss, elapsed):
    return {
        "type": "epoch",
        "epoch": n,
        "train_loss": 0.5,
        "val_loss": val_loss,
        "val_intent_acc": 0.99,
        "val_slot_f1": 0.98,
        "elapsed": elapsed,
    }


def test_a_live_run_reports_progress_speed_and_time_left(runs):
    f = runs / "runs" / "v9" / "metrics.jsonl"
    write(
        f,
        [META, *steps(1, 100, 1), epoch(1, 0.4, 12.0), *steps(101, 150, 2, t0=2.0)],
        trailing='{"type": "step", "half-wr',
    )  # a line still being written
    s = run_metrics.summarize("v9")
    assert s["status"] == "running"
    assert (s["epoch"], s["epochs_done"], s["step"], s["step_in_epoch"]) == (2, 1, 150, 50)
    assert s["rate"] == pytest.approx(10.0)  # 0.1 s per step
    # 150 steps left at 10/s, plus two validations of ~2 s each.
    assert s["eta"] == pytest.approx(150 / 10 + 2 * 2.0, rel=0.05)
    assert s["best_epoch"] == 1 and s["epochs_since_best"] == 0


def test_new_lines_are_read_incrementally_with_the_smoothed_loss(runs):
    f = runs / "runs" / "v9" / "metrics.jsonl"
    write(f, [META, *steps(1, 10, 1)])
    first = run_metrics.snapshot("v9")
    offset = first["summary"]["offset"]
    assert first["steps"][-1]["ema"] is not None
    write(f, steps(11, 12, 1))
    records, new_offset = run_metrics.tail("v9", offset)
    assert [r["step"] for r in records] == [11, 12]
    assert all("ema" in r for r in records) and new_offset > offset
    assert run_metrics.tail("v9", new_offset) == ([], new_offset)


def test_lines_written_while_tailing_wait_for_their_smoothed_loss(runs, monkeypatch):
    f = runs / "runs" / "v9" / "metrics.jsonl"
    write(f, [META, *steps(1, 10, 1)])
    offset = run_metrics.snapshot("v9")["summary"]["offset"]
    write(f, steps(11, 12, 1))
    load = run_metrics._load

    def trainer_writes_meanwhile(name):
        got = load(name)
        write(f, steps(13, 14, 1))  # after the cache read, before tail reads on
        return got

    monkeypatch.setattr(run_metrics, "_load", trainer_writes_meanwhile)
    records, offset = run_metrics.tail("v9", offset)
    assert [(r["step"], "ema" in r) for r in records] == [(11, True), (12, True)]
    monkeypatch.setattr(run_metrics, "_load", load)
    records, _ = run_metrics.tail("v9", offset)
    assert [(r["step"], "ema" in r) for r in records] == [(13, True), (14, True)]


def test_a_rerun_that_truncates_the_file_asks_for_a_fresh_snapshot(runs):
    f = runs / "runs" / "v9" / "metrics.jsonl"
    write(f, [META, *steps(1, 50, 1)])
    offset = run_metrics.snapshot("v9")["summary"]["offset"]
    f.write_text(json.dumps(META) + "\n")
    assert run_metrics.tail("v9", offset) == ([], -1)
    assert run_metrics.summarize("v9")["step"] == 0


@pytest.mark.parametrize(
    ("end", "status"),
    [
        ({"type": "done", "best_val_loss": 0.4, "stopped_early": True, "elapsed": 20}, "finished"),
        ({"type": "error", "message": "RuntimeError: MPS out of memory", "elapsed": 5}, "failed"),
        ({"type": "stopped", "elapsed": 5}, "stopped"),
    ],
)
def test_end_records_set_the_status(runs, end, status):
    write(runs / "runs" / "v9" / "metrics.jsonl", [META, *steps(1, 5, 1), end])
    s = run_metrics.summarize("v9")
    assert s["status"] == status and s["eta"] is None
    if status == "failed":
        assert "out of memory" in s["error"]


def test_a_run_that_went_quiet_without_an_end_record_is_interrupted(runs):
    f = runs / "runs" / "v9" / "metrics.jsonl"
    write(f, [META, *steps(1, 5, 1)])
    old = time.time() - run_metrics.STALE_SEC - 5
    os.utime(f, (old, old))
    assert run_metrics.summarize("v9")["status"] == "interrupted"


def test_old_runs_get_their_validation_time_from_the_gaps(runs):
    # v11-v17 streams have no val_seconds: the gap between an epoch's last
    # step and its epoch record is used.
    write(
        runs / "runs" / "v9" / "metrics.jsonl",
        [META, *steps(1, 100, 1), epoch(1, 0.4, 13.0), *steps(101, 110, 2, t0=3.0)],
    )
    assert run_metrics._val_seconds(run_metrics._load("v9")[0]) == pytest.approx([3.0])


@pytest.mark.parametrize("bad", ["../etc", "..", "v9/../../x", "", "a" * 80, "nope"])
def test_names_that_are_not_runs(runs, bad):
    assert run_metrics.run_path(bad) is None
    assert run_metrics.summarize(bad) is None


def test_runs_are_listed_newest_version_first(runs):
    for v in ("v2", "v10", "scratch"):
        write(runs / "runs" / v / "metrics.jsonl", [{**META, "version": v}])
    assert [r["run"] for r in run_metrics.list_runs()] == ["v10", "v2", "scratch"]


def test_the_step_curve_is_downsampled_but_keeps_the_last_step(runs):
    write(runs / "runs" / "v9" / "metrics.jsonl", [META, *steps(1, 3000, 1)])
    snap = run_metrics.snapshot("v9", max_points=500)
    assert len(snap["steps"]) <= 501 and snap["steps"][-1]["step"] == 3000


def test_latest_run_metrics(runs):
    assert run_metrics.latest_run_metrics() == {"waiting": True}
    write(
        runs / "runs" / "v9" / "metrics.jsonl",
        [
            META,
            *steps(1, 20, 1),
            epoch(1, 0.5, 3.0),
            {"type": "done", "best_val_loss": 0.5, "elapsed": 3},
        ],
    )
    d = run_metrics.latest_run_metrics()
    assert d["run"] == "v9" and d["meta"]["version"] == "v9"
    assert len(d["epochs"]) == 1 and d["done"]["best_val_loss"] == 0.5


# ---------------------------------------------------------------------------
# Evals catalogue
# ---------------------------------------------------------------------------
GATE = {
    "candidate": "v9",
    "baseline": "v8",
    "gates": {
        "1 new phrasings": {"pass": True, "detail": "x"},
        "6 latency": {"pass": False, "detail": "y"},
    },
    "new_phrasings": {"candidate": {"_all": {"n": 10, "intent_acc": 0.99, "slot_f1": 0.98}}},
    "real_voice": {"candidate": {"n": 5, "right": 4, "rejected": 1, "wrong": 0}},
    "latency_p95_ms": 7.5,
}


def test_evals_list_summarizes_each_versions_gate(runs):
    m = runs / "models"
    (m / "v9").mkdir()
    (m / "v9" / "gate.json").write_text(json.dumps(GATE))
    (m / "v9" / "train_manifest.json").write_text(
        json.dumps({"row_count": 100, "intent_counts": {"a": 1}})
    )
    (m / "v8").mkdir()
    rows = evals.list_evals()
    assert [r["version"] for r in rows] == ["v9", "v8"]
    g = rows[0]["gate"]
    assert (g["passed"], g["total"], g["new_phrasings_intent"], g["real_voice"]["right"]) == (
        1,
        2,
        0.99,
        4,
    )
    assert rows[0]["rows"] == 100 and rows[1]["gate"] is None


def test_eval_detail_and_bad_versions(runs):
    (runs / "models" / "v9").mkdir()
    (runs / "models" / "v9" / "gate.json").write_text(json.dumps(GATE))
    assert evals.eval_detail("v9")["gate"]["baseline"] == "v8"
    for bad in ("v99", "../v9", "nine", ""):
        assert evals.eval_detail(bad) is None


def test_a_run_named_for_averaging_counts_as_its_version(runs):
    # v26-a, v26-b, v26-c are averaged into v26 (ML/average_models.py).
    for v in ("v24", "v26-a", "scratch"):
        write(runs / "runs" / v / "metrics.jsonl", [{**META, "version": v}])
    assert [r["run"] for r in run_metrics.list_runs()] == ["v26-a", "v24", "scratch"]
