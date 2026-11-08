"""Every model version's evaluation record, for the dashboard's Evals tab.

Sources, all written next to the weights:
- ``gate.json``: ``scripts/model_gate.py``, the six gates against a baseline,
  with per-intent scores, real voice per flight, and the examples that failed;
- ``onnx/parity.json``: ``scripts/backend_parity.py``;
- ``train_manifest.json``: dataset, hyperparameters, best epoch;
- the run's metrics stream, through ``run_metrics``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import paths, run_metrics

_VERSION = re.compile(r"^v\d{1,4}$")


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def version_path(version: str) -> Path | None:
    if not _VERSION.match(version or ""):
        return None
    d = paths.MODELS_ROOT / version
    return d if d.is_dir() else None


def _gate_summary(gate: dict[str, Any]) -> dict[str, Any]:
    gates = gate.get("gates", {})

    def cand(section: str) -> dict[str, Any]:
        return (gate.get(section) or {}).get("candidate") or {}

    phrasings = cand("new_phrasings").get("_all") or {}
    old = cand("old_data_actions")
    voice = cand("real_voice")
    negatives = cand("negatives")
    return {
        "baseline": gate.get("baseline"),
        "passed": sum(1 for g in gates.values() if g.get("pass")),
        "total": len(gates),
        "gates": {name: bool(g.get("pass")) for name, g in gates.items()},
        "new_phrasings_intent": phrasings.get("intent_acc"),
        "new_phrasings_slot_f1": phrasings.get("slot_f1"),
        "old_data_right_rate": old.get("right_rate"),
        "old_data_wrong": old.get("wrong"),
        "real_voice": {k: voice.get(k) for k in ("n", "right", "rejected", "wrong")},
        "negatives_rejected": negatives.get("rejected"),
        "latency_p95_ms": gate.get("latency_p95_ms"),
    }


def _run_summary(version: str) -> dict[str, Any] | None:
    s = run_metrics.summarize(version)
    if s is None:
        return None
    keys = (
        "status",
        "device",
        "epochs_done",
        "planned_epochs",
        "elapsed",
        "best_epoch",
        "best_val_loss",
        "val_intent_acc",
        "val_slot_f1",
        "stopped_early",
    )
    return {k: s.get(k) for k in keys}


def list_evals() -> list[dict[str, Any]]:
    """One row per model version, newest first."""
    out = []
    for v in paths.list_model_versions():
        d = Path(v["path"])
        manifest = _read(d / "train_manifest.json") or {}
        gate = _read(d / "gate.json")
        parity = _read(d / "onnx" / "parity.json")
        hp = manifest.get("hyperparams") or {}
        out.append(
            {
                "version": v["version"],
                "version_num": v["version_num"],
                "created": manifest.get("created_utc"),
                "dataset": (manifest.get("dataset_path") or "").split("/")[-1] or None,
                "rows": manifest.get("row_count"),
                "intents": len(manifest.get("intent_counts") or {}) or None,
                "best_val_loss": hp.get("best_val_loss"),
                "best_epoch": hp.get("best_epoch"),
                "gate": _gate_summary(gate) if gate else None,
                "parity": None if parity is None else {"pass": parity.get("pass")},
                "run": _run_summary(v["version"]),
            }
        )
    return out


def eval_detail(version: str) -> dict[str, Any] | None:
    d = version_path(version)
    if d is None:
        return None
    gate = _read(d / "gate.json")
    return {
        "version": version,
        "manifest": _read(d / "train_manifest.json"),
        "gate": gate,
        "gate_summary": _gate_summary(gate) if gate else None,
        "parity": _read(d / "onnx" / "parity.json"),
        "run": _run_summary(version),
    }
