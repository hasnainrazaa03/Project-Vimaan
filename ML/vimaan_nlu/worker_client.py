"""Plugin-side handle on the voice worker process (``voice_worker.py``).

Everything here runs on X-Plane's main thread, so nothing may block: replies
are read from a non-blocking pipe, requests are a few hundred bytes. The
worker's stderr (library warnings, tracebacks, faulthandler dumps) goes to a
log file so a crashed worker can be diagnosed.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class WorkerClient:
    def __init__(
        self,
        argv: Sequence[str],
        log_path: str | os.PathLike[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.argv = list(argv)
        self.log_path = Path(log_path)
        self.env = env
        self.cwd = cwd
        self.proc: subprocess.Popen[bytes] | None = None
        self._buf = b""
        self._log_fh: Any = None

    def start(self) -> None:
        self._log_fh = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_fh,
            env=self.env,
            cwd=self.cwd,
            bufsize=0,
        )
        assert self.proc.stdout is not None
        os.set_blocking(self.proc.stdout.fileno(), False)
        self._buf = b""

    @property
    def pid(self) -> int | None:
        return None if self.proc is None else self.proc.pid

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def send(self, msg: dict[str, Any]) -> bool:
        """Write one request. Returns False if the worker is gone."""
        if not self.alive:
            return False
        assert self.proc is not None and self.proc.stdin is not None
        try:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def poll(self) -> list[dict[str, Any]]:
        """Every complete reply available right now (never blocks)."""
        if self.proc is None or self.proc.stdout is None:
            return []
        fd = self.proc.stdout.fileno()
        while True:
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                break
            except OSError:
                break
            if not chunk:
                break
            self._buf += chunk
        out = []
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    out.append(
                        {"type": "log", "level": "warning", "msg": f"bad reply {line[:200]!r}"}
                    )
        return out

    def log_tail(self, lines: int = 20) -> list[str]:
        try:
            return self.log_path.read_text(errors="replace").splitlines()[-lines:]
        except OSError:
            return []

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the worker to quit (closing stdin also makes it exit); kill if stuck."""
        if self.proc is None:
            return
        self.send({"op": "quit"})
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout)
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
