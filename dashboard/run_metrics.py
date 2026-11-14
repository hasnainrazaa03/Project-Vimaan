"""Read the trainer's live metrics streams for the dashboard.

``ML/train_nlu_model.py`` appends one JSON object per line to
``ML/training_runs/<version>/metrics.jsonl``: ``meta`` once, then ``step``
records, ``phase`` markers (validate / save), one ``epoch`` record per epoch,
and finally ``done`` (or ``error`` / ``stopped``). A run is watched the same
way however it was started (from the dashboard or from a terminal), since
everything comes from that file.

Each file is parsed incrementally: a cache keeps the byte offset already
read, so a live run costs only its new lines per request.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths

MAX_STEP_POINTS = 1500  # downsampled step curve: enough to look continuous
EMA_ALPHA = 0.02  # smoothing for the loss curve (about a 100-step window)
RATE_WINDOW = 200  # steps used for the current speed
STALE_SEC = 180.0  # no write this long and no end record: not live any more
LOG_TAIL_LINES = 150

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PROGRESS_NOISE = re.compile(r"%\||\bit/s\b|\bexample/s\b")


@dataclass
class _Run:
    offset: int = 0
    size: int = 0
    meta: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    epochs: list[dict[str, Any]] = field(default_factory=list)
    phase: dict[str, Any] | None = None
    end: dict[str, Any] | None = None  # done / error / stopped
    ema: float | None = None
    ema_n: int = 0


_cache: dict[str, _Run] = {}
_lock = threading.Lock()


def run_path(name: str) -> Path | None:
    """The run's directory, or None when ``name`` isn't a run. Names come from
    the URL, so they are matched against a strict pattern and must resolve
    inside the runs folder."""
    if not _RUN_NAME.match(name or ""):
        return None
    root = paths.TRAINING_RUNS.resolve()
    d = (paths.TRAINING_RUNS / name).resolve()
    if d.parent != root or not (d / "metrics.jsonl").is_file():
        return None
    return d


def read_records(
    path: Path, offset: int = 0, end: int | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Complete JSON lines from ``offset`` on (up to byte ``end``, if given),
    and the offset after the last complete line (a half-written trailing line
    is left for next time)."""
    with open(path, "rb") as fh:
        fh.seek(offset)
        chunk = fh.read() if end is None else fh.read(max(0, end - offset))
    end = chunk.rfind(b"\n")
    if end < 0:
        return [], offset
    records = []
    for raw in chunk[: end + 1].splitlines():
        try:
            rec = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records, offset + end + 1


def _apply(run: _Run, rec: dict[str, Any]) -> dict[str, Any]:
    kind = rec.get("type")
    if kind == "meta":
        run.meta = rec
    elif kind == "step":
        loss = rec.get("loss")
        if isinstance(loss, (int, float)):
            run.ema = loss if run.ema is None else EMA_ALPHA * loss + (1 - EMA_ALPHA) * run.ema
            run.ema_n += 1
            rec = {**rec, "ema": run.ema}
        run.steps.append(rec)
        run.phase = None
    elif kind == "epoch":
        run.epochs.append(rec)
        run.phase = None
    elif kind == "phase":
        run.phase = rec
    elif kind in ("done", "error", "stopped"):
        run.end = rec
    return rec


def _load(name: str) -> tuple[_Run, Path] | None:
    d = run_path(name)
    if d is None:
        return None
    f = d / "metrics.jsonl"
    size = f.stat().st_size
    with _lock:
        run = _cache.get(str(f))
        if run is None or size < run.size:  # new file, or a rerun truncated it
            run = _Run()
            _cache[str(f)] = run
        if size > run.offset:
            records, run.offset = read_records(f, run.offset)
            for rec in records:
                _apply(run, rec)
        run.size = size
    return run, d


def _status(run: _Run, d: Path) -> str:
    if run.end is not None:
        return {"done": "finished", "error": "failed", "stopped": "stopped"}[run.end["type"]]
    age = time.time() - (d / "metrics.jsonl").stat().st_mtime
    return "running" if age < STALE_SEC else "interrupted"


def _val_seconds(run: _Run) -> list[float]:
    """Validation time per epoch: logged by newer trainers, else the gap
    between an epoch's last step and its epoch record."""
    out = []
    last_step_at: dict[int, float] = {}
    for s in run.steps:
        last_step_at[s.get("epoch", 0)] = s.get("elapsed", 0.0)
    for e in run.epochs:
        if "val_seconds" in e:
            out.append(float(e["val_seconds"]))
        elif e.get("epoch") in last_step_at:
            out.append(max(0.0, e.get("elapsed", 0.0) - last_step_at[e["epoch"]]))
    return out


def summarize(name: str) -> dict[str, Any] | None:
    loaded = _load(name)
    if loaded is None:
        return None
    run, d = loaded
    meta = run.meta or {}
    status = _status(run, d)
    spe = int(meta.get("steps_per_epoch") or 0)
    planned = int(meta.get("epochs") or 0)
    last = run.steps[-1] if run.steps else None
    step = int(last["step"]) if last else 0
    epoch = int(last["epoch"]) if last else 0
    elapsed = (run.end or {}).get("elapsed") or (last or {}).get("elapsed") or 0.0

    rate = None
    if len(run.steps) >= 2:
        # Training speed within one epoch: a window across an epoch boundary
        # would count the validation pause as slow steps.
        tail_steps = run.steps[-(RATE_WINDOW + 50) :]
        window = [s for s in tail_steps if s.get("epoch") == epoch][-RATE_WINDOW:]
        if len(window) < 10:
            window = [s for s in tail_steps if s.get("epoch") == epoch - 1][-RATE_WINDOW:] or window
        if len(window) >= 2:
            dt = window[-1].get("elapsed", 0) - window[0].get("elapsed", 0)
            if dt > 0:
                rate = (window[-1]["step"] - window[0]["step"]) / dt

    eta = None
    if status == "running" and rate and spe and planned:
        remaining_steps = max(0, planned * spe - step)
        val = _val_seconds(run)
        per_val = sum(val) / len(val) if val else 0.0
        remaining_vals = max(0, planned - len(run.epochs))
        eta = remaining_steps / rate + remaining_vals * per_val

    best = min(run.epochs, key=lambda e: e.get("val_loss", float("inf")), default=None)
    latest = run.epochs[-1] if run.epochs else None
    since_best = 0
    if best is not None and latest is not None:
        since_best = int(latest["epoch"]) - int(best["epoch"])
    phase = "validating" if run.phase and run.phase.get("phase") == "validate" else None
    if run.phase and run.phase.get("phase") == "save":
        phase = "saving"

    return {
        "run": name,
        "status": status,
        "phase": phase if status == "running" else None,
        "device": meta.get("device"),
        "dataset": meta.get("dataset"),
        "version": meta.get("version", name),
        "split": meta.get("split"),
        "planned_epochs": planned,
        "steps_per_epoch": spe,
        "patience": meta.get("patience"),
        "batch_size": meta.get("batch_size"),
        "lr": meta.get("lr"),
        "max_length": meta.get("max_length"),
        "epoch": epoch,
        "epochs_done": len(run.epochs),
        "step": step,
        "step_in_epoch": step - (epoch - 1) * spe if spe and epoch else step,
        "elapsed": elapsed,
        "rate": rate,
        "eta": eta,
        "best_epoch": best and best.get("epoch"),
        "best_val_loss": best and best.get("val_loss"),
        "val_intent_acc": latest and latest.get("val_intent_acc"),
        "val_slot_f1": latest and latest.get("val_slot_f1"),
        "epochs_since_best": since_best,
        "stopped_early": (run.end or {}).get("stopped_early"),
        "error": (run.end or {}).get("message"),
        "updated": (d / "metrics.jsonl").stat().st_mtime,
        "offset": run.offset,
    }


def _downsample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    k = len(rows) / limit
    idx = sorted({int(i * k) for i in range(limit)} | {len(rows) - 1})
    return [rows[i] for i in idx]


def _log_tail(d: Path) -> list[str]:
    """The run's train.log (a terminal run's redirected output), last lines,
    without tqdm progress-bar noise."""
    f = d / "train.log"
    if not f.is_file():
        return []
    with open(f, "rb") as fh:
        fh.seek(max(0, f.stat().st_size - 96_000))
        text = fh.read().decode("utf-8", "replace")
    lines = [ln for ln in re.split(r"[\r\n]+", text) if ln.strip()]
    return [ln for ln in lines if not _PROGRESS_NOISE.search(ln)][-LOG_TAIL_LINES:]


def step_point(s: dict[str, Any]) -> dict[str, Any]:
    point = {
        "step": s["step"],
        "epoch": s.get("epoch"),
        "loss": s.get("loss"),
        "t": s.get("elapsed"),
    }
    if "ema" in s:
        point["ema"] = s["ema"]
    if "grad_norm" in s:
        point["grad_norm"] = s["grad_norm"]
    return point


def snapshot(name: str, max_points: int = MAX_STEP_POINTS) -> dict[str, Any] | None:
    """Everything the live view draws, with the step curve downsampled."""
    summary = summarize(name)
    if summary is None:
        return None
    run, d = _load(name)  # type: ignore[misc]  # summarize() just loaded it
    with _lock:
        steps = list(run.steps)
        epochs = list(run.epochs)
        meta = run.meta
    grads = [step_point(s) for s in steps if "grad_norm" in s]
    return {
        "summary": summary,
        "meta": meta,
        "steps": [step_point(s) for s in _downsample(steps, max_points)],
        "grad_norms": _downsample(grads, max_points),
        "epochs": epochs,
        "log_tail": _log_tail(d),
    }


def tail(name: str, offset: int) -> tuple[list[dict[str, Any]], int] | None:
    """New records after ``offset`` (with the EMA filled in on steps), and the
    new offset. A smaller file than ``offset`` means a rerun: the caller
    should take a fresh snapshot."""
    loaded = _load(name)
    if loaded is None:
        return None
    run, d = loaded
    with _lock:
        size, cached = run.size, run.offset
    if offset > size:
        return [], -1
    # Only as far as the cache has read: the trainer appends every ~75 ms, and
    # lines written after the cache read had no EMA, so the live smoothed curve
    # skipped them.
    records, new_offset = read_records(d / "metrics.jsonl", offset, cached)
    # The cache applied these already; take their EMA from it.
    with _lock:
        by_step = {s["step"]: s.get("ema") for s in run.steps[-(len(records) + 5) :]}
    out = []
    for rec in records:
        if rec.get("type") == "step" and rec.get("step") in by_step:
            rec = {**rec, "ema": by_step[rec["step"]]}
        out.append(rec)
    return out, new_offset


def list_runs() -> list[dict[str, Any]]:
    """Every run with a metrics stream, newest version first."""
    if not paths.TRAINING_RUNS.is_dir():
        return []
    out = []
    for d in paths.TRAINING_RUNS.iterdir():
        if d.is_dir() and (d / "metrics.jsonl").is_file():
            s = summarize(d.name)
            if s is not None:
                out.append(s)

    def order(s: dict[str, Any]) -> tuple[int, float]:
        # "v26-a" (a run to be averaged into v26) counts as 26, newest first.
        m = re.match(r"v(\d+)", s["run"])
        return (int(m.group(1)) if m else -1, s["updated"])

    out.sort(key=order, reverse=True)
    return out


def latest_run_metrics() -> dict[str, Any]:
    """The most recently written run, in the shape the old view expected."""
    runs = list_runs()
    if not runs:
        return {"waiting": True}
    newest = max(runs, key=lambda s: s["updated"])
    snap = snapshot(newest["run"]) or {}
    return {
        "run": newest["run"],
        "meta": snap.get("meta"),
        "steps": snap.get("steps", []),
        "epochs": snap.get("epochs", []),
        "done": None if newest["status"] == "running" else newest,
        "summary": newest,
    }
