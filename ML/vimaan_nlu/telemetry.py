"""Structured runtime telemetry for the X-Plane plugin.

Answers three questions from the logs alone, without a debugger attached:

* **What happened?** Every push-to-talk gets a command id (``c0007``) that
  tags each stage it passes through: capture -> STT -> NLU -> decision ->
  sim writes -> done. Grep one id to follow one command end to end.
* **Is it slow?** Each stage is timed; stages over their budget are logged at
  WARNING with ``slow: true``. A flight-loop monitor flags sim stalls (gaps
  between our callbacks) and main-thread time we spend inside X-Plane's frame.
* **Did it crash or hang?** ``faulthandler`` writes native-crash tracebacks
  (e.g. a segfault in torch) to a faults file, uncaught exceptions on any
  thread are recorded, and a hang detector dumps every thread's stack if the
  main thread stops calling back. A session with no ``plugin_stop`` event
  ended abruptly.

Output per session, under ``~/Vimaan_Logs/``:

* ``vimaan_<ts>.log``          human-readable (also mirrored to X-Plane's log)
* ``vimaan_<ts>.events.jsonl`` one JSON object per event, flushed per line so
                               a crash never loses what came before it
* ``vimaan_<ts>.faults.log``   native crash / hang stack dumps

``scripts/vimaan_log_report.py`` summarises a session from these files.

Pure standard library and free of ``xp`` / torch, so it is unit-testable and
safe to import first.
"""

from __future__ import annotations

import faulthandler
import itertools
import json
import logging
import os
import platform
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

# Per-stage latency budgets (ms). Exceeding one logs a WARNING tagged slow.
STAGE_BUDGET_MS: dict[str, float] = {
    "stt": 3_000.0,
    "nlu": 150.0,
    "dispatch": 30.0,
    "model_load": 30_000.0,
    "self_test": 5_000.0,
}
# End-to-end budget from end of speech to the sim being changed.
COMMAND_BUDGET_MS = 4_000.0
# Main-thread time we may spend per flight-loop callback before it shows as
# a frame hitch (one frame at 60 fps is ~16 ms).
MAIN_THREAD_BUDGET_MS = 20.0
# A gap this long between our flight-loop callbacks means X-Plane stalled
# (loading scenery, a modal dialog, or a hang somewhere in the sim).
STALL_GAP_SEC = 2.0
# If the main thread stops calling back for this long, dump all stacks.
HANG_DUMP_SEC = 30.0
HEARTBEAT_SEC = 60.0


def _peak_rss_mb() -> float | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    # macOS reports bytes, Linux kilobytes.
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return repr(value)


class Telemetry:
    """Session event log + timers + crash/hang/stall detection."""

    def __init__(
        self,
        log_dir: str | os.PathLike[str] | None = None,
        *,
        logger: logging.Logger | None = None,
        mirror: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        install_crash_handlers: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir else Path.home() / "Vimaan_Logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = stamp
        self.log_path = self.log_dir / f"vimaan_{stamp}.log"
        self.events_path = self.log_dir / f"vimaan_{stamp}.events.jsonl"
        self.faults_path = self.log_dir / f"vimaan_{stamp}.faults.log"

        self._clock = clock
        self._t0 = clock()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._events: TextIO = open(self.events_path, "a", encoding="utf-8", buffering=1)
        self._faults: TextIO | None = None
        self._mirror = mirror

        self.logger = logger or self._make_logger()
        self.counters: dict[str, int] = {
            "commands": 0,
            "errors": 0,
            "slow": 0,
            "stalls": 0,
            "main_thread_overruns": 0,
        }
        self._last_tick: float | None = None
        self._last_heartbeat = self._t0
        self._hang_armed = False

        if install_crash_handlers:
            self._install_crash_handlers()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _make_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"vimaan.{self.session_id}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        return logger

    def _install_crash_handlers(self) -> None:
        self._faults = open(self.faults_path, "a", encoding="utf-8", buffering=1)
        # Native crashes (SIGSEGV/SIGBUS/SIGABRT in torch, PortAudio, ...)
        # write every thread's Python stack here before the process dies.
        faulthandler.enable(file=self._faults, all_threads=True)

        prev_sys_hook = self._prev_sys_hook = sys.excepthook

        def _sys_hook(exc_type, exc, tb):
            self.error("uncaught_exception", exc, thread="main", tb=tb)
            prev_sys_hook(exc_type, exc, tb)

        sys.excepthook = _sys_hook

        prev_thread_hook = self._prev_thread_hook = threading.excepthook

        def _thread_hook(args):
            if args.exc_value is not None:
                name = args.thread.name if args.thread else "?"
                self.error("uncaught_exception", args.exc_value, thread=name, tb=args.exc_traceback)
            prev_thread_hook(args)

        threading.excepthook = _thread_hook

    # ------------------------------------------------------------------
    # Core event API
    # ------------------------------------------------------------------
    def elapsed(self) -> float:
        """Seconds since the session started."""
        return self._clock() - self._t0

    def new_command_id(self) -> str:
        return f"c{next(self._ids):04d}"

    def event(self, kind: str, level: int = logging.INFO, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "t": round(self.elapsed(), 3),
            "kind": kind,
            "thread": threading.current_thread().name,
            **{k: _jsonable(v) for k, v in fields.items()},
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            try:
                self._events.write(line + "\n")
            except ValueError:  # file closed during shutdown
                pass
        summary = " ".join(f"{k}={v}" for k, v in fields.items() if k != "traceback")
        self.log(f"[{kind}] {summary}".rstrip(), level)
        if "traceback" in fields:
            self.log(str(fields["traceback"]), level)
        return record

    def log(self, message: str, level: int = logging.INFO) -> None:
        self.logger.log(level, message)
        if self._mirror is not None:
            try:
                self._mirror(f"[Vimaan] {message}")
            except Exception:
                pass

    def error(self, kind: str, exc: BaseException, tb: Any = None, **fields: Any) -> None:
        self.counters["errors"] += 1
        tb_text = "".join(
            traceback.format_exception(type(exc), exc, tb if tb is not None else exc.__traceback__)
        )
        self.event(
            kind,
            logging.ERROR,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=tb_text,
            **fields,
        )

    @contextmanager
    def stage(
        self,
        name: str,
        cmd: str | None = None,
        expected: tuple[type[BaseException], ...] = (),
        **fields: Any,
    ) -> Iterator[dict[str, Any]]:
        """Time a stage. Yields a dict the caller can add result fields to.

        On exception the stage is recorded as failed (with traceback) and the
        exception re-raised, so callers keep their own error handling.
        Exceptions in ``expected`` (e.g. "no speech before the timeout") are
        normal outcomes: logged as ``stage_ended`` without a traceback.
        """
        result: dict[str, Any] = {}
        start = self._clock()
        try:
            yield result
        except BaseException as exc:
            ms = round((self._clock() - start) * 1000, 1)
            if expected and isinstance(exc, expected):
                self.event(
                    "stage_ended", stage=name, cmd=cmd, ms=ms, reason=type(exc).__name__, **fields
                )
            else:
                self.error("stage_failed", exc, stage=name, cmd=cmd, ms=ms, **fields)
            raise
        ms = (self._clock() - start) * 1000
        budget = STAGE_BUDGET_MS.get(name)
        slow = budget is not None and ms > budget
        if slow:
            self.counters["slow"] += 1
        self.event(
            "stage",
            logging.WARNING if slow else logging.INFO,
            stage=name,
            cmd=cmd,
            ms=round(ms, 1),
            **({"slow": True, "budget_ms": budget} if slow else {}),
            **fields,
            **result,
        )

    def command_done(self, cmd: str, outcome: str, started_at: float, **fields: Any) -> None:
        """Close out one command. ``started_at`` is a ``clock()`` reading."""
        self.counters["commands"] += 1
        total = (self._clock() - started_at) * 1000
        slow = total > COMMAND_BUDGET_MS
        self.event(
            "command_done",
            logging.WARNING if slow or outcome not in ("executed", "no_op") else logging.INFO,
            cmd=cmd,
            outcome=outcome,
            total_ms=round(total, 1),
            **({"slow": True} if slow else {}),
            **fields,
        )

    # ------------------------------------------------------------------
    # Main-thread health
    # ------------------------------------------------------------------
    def tick(self, busy_ms: float | None = None, **heartbeat_fields: Any) -> None:
        """Call once per flight-loop callback (main thread).

        Detects sim stalls from the gap since the previous tick, flags our own
        main-thread overruns, re-arms the hang detector and emits a periodic
        heartbeat.
        """
        now = self._clock()
        if self._last_tick is not None:
            gap = now - self._last_tick
            if gap > STALL_GAP_SEC:
                self.counters["stalls"] += 1
                self.event("sim_stall", logging.WARNING, gap_ms=round(gap * 1000))
        self._last_tick = now

        if busy_ms is not None and busy_ms > MAIN_THREAD_BUDGET_MS:
            self.counters["main_thread_overruns"] += 1
            self.event(
                "main_thread_overrun",
                logging.WARNING,
                ms=round(busy_ms, 1),
                budget_ms=MAIN_THREAD_BUDGET_MS,
            )

        self.pet_hang_detector()

        if now - self._last_heartbeat >= HEARTBEAT_SEC:
            self._last_heartbeat = now
            self.event(
                "heartbeat",
                logging.DEBUG,
                peak_rss_mb=_peak_rss_mb(),
                threads=threading.active_count(),
                **self.counters,
                **heartbeat_fields,
            )

    def pet_hang_detector(self) -> None:
        """(Re)arm the hang dump. If not petted within HANG_DUMP_SEC, every
        thread's stack is written to the faults file (process keeps running)."""
        if self._faults is None:
            return
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(HANG_DUMP_SEC, repeat=False, file=self._faults)
        self._hang_armed = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def environment(self) -> dict[str, Any]:
        """Facts about the runtime worth having in every bug report."""
        info: dict[str, Any] = {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pid": os.getpid(),
            "env": {k: v for k, v in os.environ.items() if k.startswith("VIMAAN_")},
        }
        for mod in ("torch", "transformers", "speech_recognition", "numpy", "onnxruntime"):
            m = sys.modules.get(mod)
            if m is not None:
                info[f"{mod}_version"] = getattr(m, "__version__", "?")
        return info

    def close(self) -> None:
        self.event("session_end", peak_rss_mb=_peak_rss_mb(), **self.counters)
        if self._hang_armed:
            faulthandler.cancel_dump_traceback_later()
            self._hang_armed = False
        if self._faults is not None:
            # Restore the hooks we chained so a plugin reload doesn't stack them.
            sys.excepthook = self._prev_sys_hook
            threading.excepthook = self._prev_thread_hook
            faulthandler.disable()
            self._faults.close()
            self._faults = None
        with self._lock:
            self._events.close()
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
