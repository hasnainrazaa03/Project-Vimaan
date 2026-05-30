"""Dashboard tests — pure-Python paths only (no torch, no real training)."""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

import pytest

# Make `dashboard` importable from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fastapi = pytest.importorskip("fastapi")  # noqa: F841
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402

from dashboard import paths as dpaths  # noqa: E402
from dashboard.log_parser import parse_line  # noqa: E402
from dashboard.training import TrainingManager  # noqa: E402

# ---- log_parser ------------------------------------------------------------


class TestLogParser:
    def test_train_loss(self):
        e = parse_line("Epoch 3 - Average Training Loss: 0.1234")
        assert e == {"type": "train_loss", "epoch": 3, "value": 0.1234}

    def test_val_loss(self):
        e = parse_line("Epoch 7 - Average Validation Loss: 0.0567")
        assert e == {"type": "val_loss", "epoch": 7, "value": 0.0567}

    def test_checkpoint_saved(self):
        e = parse_line("Validation loss improved! Saving best model to /a/b/v10")
        assert e == {"type": "checkpoint_saved", "path": "/a/b/v10"}

    def test_no_improve(self):
        e = parse_line("Validation loss did not improve. Count: 1/2")
        assert e == {"type": "no_improve", "count": 1, "patience": 2}

    def test_early_stop(self):
        e = parse_line("Early stopping triggered after 6 epochs.")
        assert e == {"type": "early_stop", "epoch": 6}

    def test_unrelated_line(self):
        assert parse_line("Loading final dataset...") is None
        assert parse_line("") is None


# ---- paths -----------------------------------------------------------------


class TestPaths:
    def test_safe_upload_path_accepts_jsonl(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dpaths, "UPLOAD_DIR", tmp_path)
        p = dpaths.safe_upload_path("my_data.jsonl")
        assert p.parent == tmp_path
        assert p.name == "my_data.jsonl"

    def test_safe_upload_path_strips_directories(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dpaths, "UPLOAD_DIR", tmp_path)
        p = dpaths.safe_upload_path("../../etc/passwd.jsonl")
        assert p.parent == tmp_path
        assert "passwd" in p.name

    @pytest.mark.parametrize("name", ["foo.txt", "", ".hidden.jsonl"])
    def test_safe_upload_path_rejects_bad_names(self, name):
        with pytest.raises(ValueError):
            dpaths.safe_upload_path(name)

    def test_list_model_versions_orders_newest_first(self, monkeypatch, tmp_path):
        for v in ("v1", "v3", "v10", "v2"):
            (tmp_path / v).mkdir()
        (tmp_path / "v10" / "train_manifest.json").write_text("{}")
        monkeypatch.setattr(dpaths, "MODELS_ROOT", tmp_path)
        rows = dpaths.list_model_versions()
        assert [r["version"] for r in rows] == ["v10", "v3", "v2", "v1"]
        assert rows[0]["has_manifest"] is True
        assert rows[1]["has_manifest"] is False


# ---- TrainingManager (fake subprocess) -------------------------------------


@pytest.fixture()
def fake_train_script(tmp_path, monkeypatch):
    """Replace TRAIN_SCRIPT with a tiny python script that emits real log lines."""
    script = tmp_path / "fake_train.py"
    script.write_text(
        textwrap.dedent("""
        import sys, time
        print("Loading final dataset...", flush=True)
        print("Epoch 1 - Average Training Loss: 0.5000", flush=True)
        print("Epoch 1 - Average Validation Loss: 0.4500", flush=True)
        print("Validation loss improved! Saving best model to /tmp/fake/v1", flush=True)
        print("Epoch 2 - Average Training Loss: 0.3000", flush=True)
        print("Epoch 2 - Average Validation Loss: 0.2500", flush=True)
        sys.exit(0)
    """)
    )
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"text":"a","intent":"x","slots":{}}\n')
    monkeypatch.setattr("dashboard.training.TRAIN_SCRIPT", script)
    monkeypatch.setattr("dashboard.training.ML_DIR", tmp_path)
    return script, ds


def test_training_manager_runs_and_collects_metrics(fake_train_script):
    _script, ds = fake_train_script
    mgr = TrainingManager()
    snap = mgr.start(str(ds), epochs=2)
    assert snap["status"] == "running"

    # Wait for the fake script to finish (it's fast).
    for _ in range(50):
        if not mgr.is_running():
            break
        time.sleep(0.1)
    else:
        mgr.stop()
        pytest.fail("training did not finish in time")

    final = mgr.snapshot()
    assert final["status"] == "finished"
    assert final["exit_code"] == 0
    assert final["metrics"] == [
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.45},
        {"epoch": 2, "train_loss": 0.3, "val_loss": 0.25},
    ]
    assert "/tmp/fake/v1" in final["checkpoints"]


def test_training_manager_rejects_concurrent_start(fake_train_script):
    _script, ds = fake_train_script
    mgr = TrainingManager()
    mgr.start(str(ds))
    try:
        with pytest.raises(RuntimeError):
            mgr.start(str(ds))
    finally:
        mgr.stop()


def test_training_manager_missing_dataset_raises(tmp_path):
    mgr = TrainingManager()
    with pytest.raises(FileNotFoundError):
        mgr.start(str(tmp_path / "nope.jsonl"))


# ---- FastAPI app (no torch) ------------------------------------------------


@pytest.fixture()
def client(monkeypatch, tmp_path):
    # Redirect paths into the temp dir BEFORE importing app, so ensure_dirs picks it up.
    monkeypatch.setattr(dpaths, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(dpaths, "DATASETS_FINAL", tmp_path / "datasets")
    monkeypatch.setattr(dpaths, "UPLOAD_DIR", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir(parents=True)
    (tmp_path / "models" / "v1").mkdir(parents=True)
    (tmp_path / "datasets").mkdir()

    # Reload the app module so route closures pick up the patched paths.
    import importlib

    import dashboard.app as app_mod

    importlib.reload(app_mod)
    return TestClient(app_mod.app)


class TestAPI:
    def test_index_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Project Vimaan" in r.text

    def test_models_list(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        body = r.json()
        assert len(body["models"]) == 1
        assert body["models"][0]["version"] == "v1"

    def test_datasets_list_empty(self, client):
        r = client.get("/api/datasets")
        assert r.status_code == 200
        assert r.json() == {"datasets": []}

    def test_upload_valid_jsonl(self, client):
        payload = "\n".join(
            [
                json.dumps(
                    {
                        "text": "set heading 270",
                        "intent": "set_heading",
                        "slots": {"heading": "270"},
                    }
                ),
                json.dumps({"text": "gear up", "intent": "set_gear", "slots": {}}),
            ]
        ).encode()
        r = client.post(
            "/api/datasets/upload",
            files={"file": ("good.jsonl", payload, "application/json")},
        )
        assert r.status_code == 200, r.text
        assert r.json()["rows"] == 2

    def test_upload_rejects_non_jsonl_extension(self, client):
        r = client.post(
            "/api/datasets/upload",
            files={"file": ("evil.txt", b"hi", "text/plain")},
        )
        assert r.status_code == 400

    def test_upload_rejects_malformed_json(self, client):
        r = client.post(
            "/api/datasets/upload",
            files={"file": ("bad.jsonl", b"not json\n", "application/json")},
        )
        assert r.status_code == 400

    def test_upload_rejects_missing_keys(self, client):
        r = client.post(
            "/api/datasets/upload",
            files={"file": ("nokeys.jsonl", b'{"foo":"bar"}\n', "application/json")},
        )
        assert r.status_code == 400

    def test_train_start_rejects_outside_repo(self, client):
        r = client.post("/api/train/start", json={"dataset": "/etc/passwd"})
        # Either "outside repo" (400) or "not found" (404) — both are OK
        # so long as we don't 200 and run something dangerous.
        assert r.status_code in (400, 404)

    def test_train_status_idle(self, client):
        r = client.get("/api/train/status")
        assert r.status_code == 200
        assert r.json()["status"] in ("idle", "finished", "failed")
