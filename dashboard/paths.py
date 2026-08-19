"""Filesystem layout helpers — kept torch-free so unit tests can import."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = REPO_ROOT / "ML"
MODELS_ROOT = ML_DIR / "models" / "vimaan_nlu_model_best"
DATASETS_ROOT = ML_DIR / "datasets"
DATASETS_FINAL = DATASETS_ROOT / "05_final_merged"
UPLOAD_DIR = DATASETS_ROOT / "dashboard_uploads"
TRAIN_SCRIPT = ML_DIR / "train_nlu_model.py"
TRAINING_RUNS = ML_DIR / "training_runs"


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def list_model_versions() -> list[dict]:
    """Return [{version, path, has_manifest, mtime}] sorted newest first."""
    if not MODELS_ROOT.exists():
        return []
    out = []
    for entry in MODELS_ROOT.iterdir():
        if not entry.is_dir() or not entry.name.startswith("v"):
            continue
        try:
            num = int(entry.name[1:])
        except ValueError:
            continue
        try:
            mtime = entry.stat().st_mtime  # may vanish between iterdir() and stat()
        except OSError:
            continue
        out.append(
            {
                "version": entry.name,
                "version_num": num,
                "path": str(entry),
                "has_manifest": (entry / "train_manifest.json").is_file(),
                "mtime": mtime,
            }
        )
    out.sort(key=lambda d: d["version_num"], reverse=True)
    return out


def list_datasets() -> list[dict]:
    """Return all .jsonl files under DATASETS_FINAL and UPLOAD_DIR."""
    out = []
    for root in (DATASETS_FINAL, UPLOAD_DIR):
        if not root.exists():
            continue
        for p in sorted(root.glob("*.jsonl")):
            try:
                size = p.stat().st_size
            except OSError:
                continue
            out.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": size,
                    "source": "uploaded" if root == UPLOAD_DIR else "merged",
                }
            )
    return out


def safe_model_path(model_path: str) -> Path:
    """Resolve a client-supplied checkpoint path and confine it to MODELS_ROOT.

    The predict endpoint loads weights from this path (``from_pretrained`` +
    ``torch.load``), so an unconstrained path is a traversal / arbitrary-load
    vector — even on localhost, where a browser can reach the dashboard via
    DNS-rebinding. Relative paths resolve under MODELS_ROOT; absolute paths
    must still land inside it. Raises ValueError otherwise.
    """
    if not model_path or not model_path.strip():
        raise ValueError("model_path is required")
    p = Path(model_path)
    if not p.is_absolute():
        p = MODELS_ROOT / model_path
    p = p.resolve()
    try:
        p.relative_to(MODELS_ROOT.resolve())
    except ValueError as e:
        raise ValueError("model_path must live inside the models directory") from e
    return p


def safe_dataset_path(dataset: str) -> Path:
    """Resolve a client-supplied dataset path and confine it to the repo.

    Absolute paths are resolved BEFORE the containment check. The earlier train
    endpoint skipped ``resolve()`` for absolute paths, so an absolute ``..``
    path (e.g. ``<repo>/../../etc/passwd``) slipped past the purely-lexical
    ``relative_to`` check. Raises ValueError on escape.
    """
    if not dataset or not dataset.strip():
        raise ValueError("dataset is required")
    p = Path(dataset)
    if not p.is_absolute():
        p = REPO_ROOT / dataset
    p = p.resolve()
    try:
        p.relative_to(REPO_ROOT.resolve())
    except ValueError as e:
        raise ValueError("dataset must live inside the repo") from e
    return p


def safe_upload_path(filename: str) -> Path:
    """Sanitise a user-supplied filename and return its target path under UPLOAD_DIR."""
    base = os.path.basename(filename).strip()
    if not base or base.startswith("."):
        raise ValueError("invalid filename")
    if not base.endswith(".jsonl"):
        raise ValueError("only .jsonl uploads are accepted")
    safe = "".join(c for c in base if c.isalnum() or c in ("-", "_", "."))
    # Re-validate after filtering: stripping non-safe chars (e.g. an emoji stem)
    # can leave a hidden/dotfile like ".jsonl".
    if not safe or safe.startswith("."):
        raise ValueError("filename has no safe characters")
    return UPLOAD_DIR / safe
