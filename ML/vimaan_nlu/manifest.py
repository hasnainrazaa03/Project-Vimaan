"""Training-run provenance manifest.

A manifest is a JSON sidecar written next to every saved model. It records
exactly which dataset + git revision + hyperparameters produced the
artifact, so a trained model can be reproduced or audited later.

This module has zero hard dependencies on torch/transformers — framework
versions are looked up via lazy import and silently fall back to "unknown"
if those packages are absent. That keeps the manifest helpers importable
in CI (which does not install torch) and in the X-Plane plugin.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "train_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def compute_dataset_sha256(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_intent_counts(rows: Iterable[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        intent = row.get("intent")
        if intent:
            counts[intent] += 1
    return dict(sorted(counts.items()))


def get_git_sha(repo_dir: str | os.PathLike | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _framework_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import torch

        versions["torch"] = getattr(torch, "__version__", "unknown")
    except Exception:
        versions["torch"] = "unknown"
    try:
        import transformers

        versions["transformers"] = getattr(transformers, "__version__", "unknown")
    except Exception:
        versions["transformers"] = "unknown"
    return versions


def build_manifest(
    *,
    dataset_path: str | os.PathLike,
    dataset_rows: Iterable[dict] | None = None,
    hyperparams: dict[str, Any] | None = None,
    row_count: int | None = None,
    intent_counts: dict[str, int] | None = None,
    git_sha: str | None = None,
    framework_versions: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the manifest dict. Callers may pass `dataset_rows` (and we'll
    compute counts) or pre-computed `row_count`/`intent_counts`."""
    if dataset_rows is not None:
        rows_list = list(dataset_rows)
        if row_count is None:
            row_count = len(rows_list)
        if intent_counts is None:
            intent_counts = compute_intent_counts(rows_list)

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "dataset_sha256": compute_dataset_sha256(dataset_path),
        "row_count": row_count if row_count is not None else 0,
        "intent_counts": intent_counts or {},
        "git_sha": git_sha if git_sha is not None else get_git_sha(),
        "framework_versions": framework_versions or _framework_versions(),
        "hyperparams": hyperparams or {},
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(model_dir: str | os.PathLike, manifest: dict[str, Any]) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / MANIFEST_FILENAME
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return out_path


def load_manifest(model_dir: str | os.PathLike) -> dict[str, Any]:
    path = Path(model_dir) / MANIFEST_FILENAME
    with open(path) as fh:
        return json.load(fh)
