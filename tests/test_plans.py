"""Training plans (dashboard/plans.py) on synthetic runs."""

import json
import os
import time

import pytest

from dashboard import paths, plans, run_metrics


@pytest.fixture
def area(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "TRAINING_RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "TRAINING_PLANS", tmp_path / "plans")
    (tmp_path / "runs").mkdir()
    (tmp_path / "plans").mkdir()
    run_metrics._cache.clear()
    return tmp_path


META = {"type": "meta", "epochs": 10, "steps_per_epoch": 100, "eval_every": 0.5}


def run(area, name, elapsed, done=True, score=None):
    path = area / "runs" / name / "metrics.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        META,
        {"type": "step", "step": 1, "epoch": 1, "loss": 1.0, "elapsed": 1.0},
        {"type": "step", "step": 50, "epoch": 1, "loss": 0.5, "elapsed": elapsed},
        {
            "type": "epoch",
            "epoch": 0.5,
            "train_loss": 0.5,
            "val_loss": 0.1,
            "best": True,
            "dev": {"score": score} if score is not None else None,
            "elapsed": elapsed,
        },
    ]
    if done:
        records.append({"type": "done", "elapsed": elapsed})
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def plan(area, name="round", **extra):
    body = {
        "title": "round 4",
        "groups": [
            {"name": "plain", "runs": ["a1", "a2", "a3"]},
            {"name": "twice the work", "runs": ["b1", "b2"], "cost": 2},
        ],
        **extra,
    }
    (area / "plans" / f"{name}.json").write_text(json.dumps(body))


def test_a_plan_shows_each_runs_state_and_the_time_left(area):
    run(area, "a1", 600, score=5900.5)
    run(area, "a2", 800)
    run(area, "a3", 100, done=False)  # training now
    plan(area)
    (p,) = plans.list_plans()
    assert (p["title"], p["total"], p["done"], p["running"]) == ("round 4", 5, 2, ["a3"])
    plain, twice = p["groups"]
    assert [r["status"] for r in plain["runs"]] == ["finished", "finished", "running"]
    assert plain["runs"][0]["best_score"] == 5900.5
    assert plain["each_seconds"] == 700  # its finished runs' mean
    assert plain["runs"][2]["left"] == 600  # 700 less the 100 it has run
    assert [r["status"] for r in twice["runs"]] == ["queued", "queued"]
    assert twice["each_seconds"] == 1400  # no finished run: the plan's, twice
    assert p["left_seconds"] == 600 + 2 * 1400
    assert p["ends_at"] == pytest.approx(time.time() + p["left_seconds"], abs=5)


def test_with_nothing_finished_the_time_left_is_unknown(area):
    plan(area)
    (p,) = plans.list_plans()
    assert p["done"] == 0 and p["left_seconds"] is None and p["ends_at"] is None


def test_a_broken_plan_is_listed_with_its_error_and_the_newest_comes_first(area):
    plan(area, "older")
    (area / "plans" / "broken.json").write_text("{not json")
    os.utime(area / "plans" / "older.json", (time.time() - 60, time.time() - 60))
    listed = plans.list_plans()
    assert [p["name"] for p in listed] == ["broken", "older"]
    assert "error" in listed[0] and "groups" in listed[1]


def test_a_run_name_that_is_not_a_run_is_only_ever_queued(area):
    plan(area, groups=[{"name": "g", "runs": ["../../etc", "ok"]}])
    (p,) = plans.list_plans()
    assert [r["status"] for r in p["groups"][0]["runs"]] == ["queued", "queued"]
