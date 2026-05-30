"""Subprocess-based training runner with in-memory log + metric buffers.

Single-flight: only one training job at a time per process. The dashboard
is intentionally single-user.
"""

from __future__ import annotations

import collections
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .log_parser import parse_line
from .paths import ML_DIR, TRAIN_SCRIPT

_MAX_LOG_LINES = 2000


@dataclass
class TrainingState:
    status: str = "idle"  # idle | running | finished | failed
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    pid: int | None = None
    dataset: str | None = None
    hyperparams: dict[str, Any] = field(default_factory=dict)
    logs: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=_MAX_LOG_LINES)
    )
    metrics: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)


class TrainingManager:
    def __init__(self) -> None:
        self._state = TrainingState()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            s = self._state
            return {
                "status": s.status,
                "started_at": s.started_at,
                "finished_at": s.finished_at,
                "exit_code": s.exit_code,
                "pid": s.pid,
                "dataset": s.dataset,
                "hyperparams": dict(s.hyperparams),
                "log_tail": list(s.logs)[-200:],
                "metrics": list(s.metrics),
                "checkpoints": list(s.checkpoints),
            }

    def is_running(self) -> bool:
        with self._lock:
            return self._state.status == "running"

    def start(
        self,
        dataset: str,
        *,
        epochs: int = 10,
        lr: float = 5e-5,
        batch_size: int = 16,
        base_model: str = "distilbert-base-uncased",
        max_length: int = 64,
        patience: int = 2,
    ) -> dict[str, Any]:
        if self.is_running():
            raise RuntimeError("a training job is already running")

        ds = Path(dataset)
        if not ds.is_file():
            raise FileNotFoundError(f"dataset not found: {dataset}")

        cmd = [
            sys.executable,
            "-u",
            str(TRAIN_SCRIPT),
            "--dataset",
            str(ds),
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--batch-size",
            str(batch_size),
            "--base-model",
            base_model,
            "--max-length",
            str(max_length),
            "--patience",
            str(patience),
        ]

        with self._lock:
            self._state = TrainingState(
                status="running",
                started_at=time.time(),
                dataset=str(ds),
                hyperparams={
                    "epochs": epochs,
                    "lr": lr,
                    "batch_size": batch_size,
                    "base_model": base_model,
                    "max_length": max_length,
                    "patience": patience,
                },
            )

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(ML_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with self._lock:
            self._state.pid = self._proc.pid

        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        return self.snapshot()

    def _pump(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            line = raw.rstrip("\n")
            event = parse_line(line)
            with self._lock:
                self._state.logs.append(line)
                if event is None:
                    continue
                if event["type"] in ("train_loss", "val_loss"):
                    self._merge_metric(event)
                elif event["type"] == "checkpoint_saved":
                    self._state.checkpoints.append(event["path"])
        rc = self._proc.wait()
        with self._lock:
            self._state.exit_code = rc
            self._state.finished_at = time.time()
            self._state.status = "finished" if rc == 0 else "failed"

    def _merge_metric(self, event: dict[str, Any]) -> None:
        """Combine train/val loss for the same epoch into one row."""
        epoch = event["epoch"]
        key = "train_loss" if event["type"] == "train_loss" else "val_loss"
        for row in self._state.metrics:
            if row["epoch"] == epoch:
                row[key] = event["value"]
                return
        self._state.metrics.append({"epoch": epoch, key: event["value"]})


MANAGER = TrainingManager()
