"""FastAPI app exposing the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .paths import (
    REPO_ROOT,
    ensure_dirs,
    list_datasets,
    list_model_versions,
    safe_upload_path,
)
from .predictor import predict as run_predict
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

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty file")
    if len(contents) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large (>200MB)")

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
        return run_predict(req.model_path, req.text)
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
    # Resolve dataset against the repo root to keep clients honest.
    ds_path = Path(req.dataset)
    if not ds_path.is_absolute():
        ds_path = (REPO_ROOT / req.dataset).resolve()
    try:
        ds_path.relative_to(REPO_ROOT)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="dataset must live inside the repo") from e

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


# ---- static (mount last so /api/* wins) ------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(404)
async def _not_found(_req, exc) -> JSONResponse:  # pragma: no cover
    return JSONResponse({"detail": "not found"}, status_code=404)
