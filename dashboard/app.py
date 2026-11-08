"""FastAPI app exposing the dashboard."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import evals, run_metrics
from .paths import (
    ensure_dirs,
    list_datasets,
    list_model_versions,
    safe_dataset_path,
    safe_model_path,
    safe_upload_path,
)
from .predictor import predict as run_predict
from .run_metrics import latest_run_metrics
from .training import MANAGER

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Project Vimaan Dashboard", version="0.1.0")
ensure_dirs()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ---- catalogue endpoints ---------------------------------------------------


@app.get("/api/models")
def api_models() -> dict[str, Any]:
    versions = list_model_versions()
    for v in versions:
        manifest_path = Path(v["path"]) / "train_manifest.json"
        if manifest_path.is_file():
            try:
                v["manifest"] = json.loads(manifest_path.read_text())
            except Exception:  # pragma: no cover - corrupted manifest
                v["manifest"] = None
    return {"models": versions}


@app.get("/api/datasets")
def api_datasets() -> dict[str, Any]:
    return {"datasets": list_datasets()}


# ---- dataset upload --------------------------------------------------------


@app.post("/api/datasets/upload")
async def api_upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        target = safe_upload_path(file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Read in chunks and abort past the cap, so an oversized upload is not fully
    # buffered into memory before the size is checked.
    max_bytes = 200 * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="file too large (>200MB)")
        chunks.append(chunk)
    contents = b"".join(chunks)
    if not contents:
        raise HTTPException(status_code=400, detail="empty file")

    # Light validation: every line must be JSON with `text` and `intent`.
    line_count = 0
    for raw in contents.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"line {line_count + 1}: invalid JSON ({e.msg})",
            ) from e
        if "text" not in row or "intent" not in row:
            raise HTTPException(
                status_code=400,
                detail=f"line {line_count + 1}: missing 'text' or 'intent' key",
            )
        line_count += 1

    if line_count == 0:
        raise HTTPException(status_code=400, detail="no non-empty lines")

    target.write_bytes(contents)
    return {"name": target.name, "path": str(target), "rows": line_count}


# ---- prediction ------------------------------------------------------------


class PredictRequest(BaseModel):
    model_path: str = Field(..., description="Path to a checkpoint directory")
    text: str = Field(..., min_length=1, max_length=2000)


@app.post("/api/predict")
def api_predict(req: PredictRequest) -> dict[str, Any]:
    try:
        model_path = safe_model_path(req.model_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        return run_predict(str(model_path), req.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModuleNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"inference deps missing ({e.name}); install torch/transformers",
        ) from e
    except Exception as e:  # pragma: no cover - surface unexpected failures
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


# ---- training --------------------------------------------------------------


class TrainRequest(BaseModel):
    dataset: str
    epochs: int = Field(10, ge=1, le=100)
    lr: float = Field(5e-5, gt=0, le=1)
    batch_size: int = Field(16, ge=1, le=512)
    base_model: str = "distilbert-base-uncased"
    max_length: int = Field(64, ge=8, le=512)
    patience: int = Field(2, ge=0, le=20)


@app.post("/api/train/start")
def api_train_start(req: TrainRequest) -> dict[str, Any]:
    # Confine the dataset to the repo (resolves absolute paths first).
    try:
        ds_path = safe_dataset_path(req.dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        return MANAGER.start(
            str(ds_path),
            epochs=req.epochs,
            lr=req.lr,
            batch_size=req.batch_size,
            base_model=req.base_model,
            max_length=req.max_length,
            patience=req.patience,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@app.post("/api/train/stop")
def api_train_stop() -> dict[str, Any]:
    return MANAGER.stop()


@app.get("/api/train/status")
def api_train_status() -> dict[str, Any]:
    return MANAGER.snapshot()


@app.get("/api/train/metrics")
def api_train_metrics() -> dict[str, Any]:
    """Rich live metrics for the newest run (split, per-step loss, per-epoch
    val intent-accuracy + slot-F1), streamed by the trainer to
    ML/training_runs/<version>/metrics.jsonl."""
    return latest_run_metrics()


# ---- training runs: live and past -----------------------------------------

STREAM_POLL_SEC = 0.5  # how often the stream checks the metrics file
STREAM_SUMMARY_SEC = 5.0  # a status refresh even with no new lines
STREAM_PING_SEC = 15.0


@app.get("/api/runs")
def api_runs() -> dict[str, Any]:
    """Every training run (dashboard or terminal), newest version first."""
    return {"runs": run_metrics.list_runs()}


@app.get("/api/runs/{name}")
def api_run(name: str, points: int = 1500) -> dict[str, Any]:
    snap = run_metrics.snapshot(name, max_points=max(50, min(points, 5000)))
    if snap is None:
        raise HTTPException(status_code=404, detail=f"no run named {name!r}")
    return snap


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/runs/{name}/stream")
async def api_run_stream(name: str, request: Request, follow: bool = True) -> StreamingResponse:
    """Server-sent events: a ``snapshot`` first, then ``records`` with each
    batch of new lines (and the updated summary), ``summary`` every few
    seconds so a stall shows up, and a fresh ``snapshot`` if the run is
    restarted. ``follow=false`` sends what there is and ends (for tests)."""
    first = run_metrics.snapshot(name)
    if first is None:
        raise HTTPException(status_code=404, detail=f"no run named {name!r}")

    async def events() -> AsyncIterator[str]:
        snap = first
        offset = snap["summary"]["offset"]
        yield _sse("snapshot", snap)
        last_summary = last_ping = time.monotonic()
        while follow:
            if await request.is_disconnected():
                return
            await asyncio.sleep(STREAM_POLL_SEC)
            got = run_metrics.tail(name, offset)
            if got is None:
                return  # the run was deleted
            records, new_offset = got
            now = time.monotonic()
            if new_offset < 0:
                snap = run_metrics.snapshot(name)
                if snap is None:
                    return
                offset = snap["summary"]["offset"]
                yield _sse("snapshot", snap)
            elif records:
                offset = new_offset
                yield _sse("records", {"records": records, "summary": run_metrics.summarize(name)})
                last_summary = last_ping = now
            elif now - last_summary >= STREAM_SUMMARY_SEC:
                yield _sse("summary", run_metrics.summarize(name))
                last_summary = last_ping = now
            elif now - last_ping >= STREAM_PING_SEC:
                yield ": ping\n\n"
                last_ping = now

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---- evaluations -----------------------------------------------------------


@app.get("/api/evals")
def api_evals() -> dict[str, Any]:
    """Every model version's gate, parity and training summary."""
    return {"versions": evals.list_evals()}


@app.get("/api/evals/{version}")
def api_eval(version: str) -> dict[str, Any]:
    detail = evals.eval_detail(version)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"no model {version!r}")
    return detail


# ---- static (mount last so /api/* wins) ------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# NOTE: no custom 404 handler — FastAPI's default already returns JSON
# ({"detail": ...}) for both unknown routes and deliberate HTTPException(404),
# and a status-code handler would clobber the detail of intentional 404s
# (e.g. "model not found").
