"""What the Mac is doing while Vimaan flies: CPU and memory of X-Plane and of
the voice worker, GPU utilisation, free memory, swap and load.

Sampled by the voice worker on its own thread (never inside X-Plane), with
the tools macOS ships and no admin rights: ``ps`` (CPU time and resident
memory per process), ``ioreg`` (the GPU's own utilisation counters),
``sysctl`` (free-memory level and swap). A tool that is missing or fails
leaves its fields out; nothing here can stop the worker.

CPU is reported as a percentage of one core over the time since the previous
sample (so 250 = two and a half cores busy), from the change in each
process's CPU time, not ``ps``'s own decaying ``%cpu``.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Callable
from typing import Any

USAGE_SEC = 15.0

Runner = Callable[[list[str]], str]


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False).stdout


def parse_cputime(text: str) -> float:
    """``ps`` CPU time to seconds: ``0:01.25``, ``12:03.50``, ``1:02:03``,
    ``2-01:02:03``."""
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [float(x) for x in text.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return days * 86400 + seconds


def parse_ps(text: str) -> dict[int, tuple[float, float]]:
    """``ps -o pid=,cputime=,rss=`` lines to {pid: (cpu seconds, resident MB)}."""
    out = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            out[int(fields[0])] = (parse_cputime(fields[1]), int(fields[2]) / 1024)
        except ValueError:
            continue
    return out


_GPU_FIELDS = {
    "Device Utilization %": "gpu_pct",
    "Renderer Utilization %": "gpu_renderer_pct",
    "Tiler Utilization %": "gpu_tiler_pct",
}


def parse_ioreg(text: str) -> dict[str, float]:
    """The GPU's utilisation counters (Apple silicon and recent Intel GPUs
    publish them in ``PerformanceStatistics``), and the memory it uses."""
    out: dict[str, float] = {}
    for key, name in _GPU_FIELDS.items():
        m = re.search(rf'"{re.escape(key)}"\s*=\s*(\d+)', text)
        if m:
            out[name] = float(m.group(1))
    m = re.search(r'"In use system memory"\s*=\s*(\d+)', text)
    if m:
        out["gpu_mem_mb"] = round(int(m.group(1)) / 2**20)
    return out


def parse_swap(text: str) -> float | None:
    """``sysctl -n vm.swapusage`` (``total = 6144.00M  used = 4763.00M ...``)
    to MB used."""
    m = re.search(r"used\s*=\s*([\d.]+)([MG])", text)
    if not m:
        return None
    return round(float(m.group(1)) * (1024 if m.group(2) == "G" else 1))


class UsageSampler:
    """One ``sample()`` = one usage record. ``pids`` names the processes to
    follow, e.g. {"xplane": the worker's parent, "worker": its own pid}."""

    def __init__(
        self,
        pids: dict[str, int],
        run: Runner = _run,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.pids = pids
        self._run = run
        self._clock = clock
        self._last: tuple[float, dict[int, float]] | None = None

    def _try(self, argv: list[str]) -> str:
        try:
            return self._run(argv)
        except Exception:
            return ""

    def sample(self) -> dict[str, Any]:
        now = self._clock()
        out: dict[str, Any] = {}
        ps = parse_ps(
            self._try(
                ["ps", "-o", "pid=,cputime=,rss=", "-p", ",".join(map(str, self.pids.values()))]
            )
        )
        cpu_now = {pid: cpu for pid, (cpu, _) in ps.items()}
        for name, pid in self.pids.items():
            if pid not in ps:
                continue
            cpu, rss = ps[pid]
            out[f"{name}_rss_mb"] = round(rss)
            if self._last is not None and pid in self._last[1] and now > self._last[0]:
                used = cpu - self._last[1][pid]
                out[f"{name}_cpu_pct"] = round(100 * used / (now - self._last[0]), 1)
        self._last = (now, cpu_now)
        out.update(
            parse_ioreg(self._try(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"]))
        )
        level = self._try(["sysctl", "-n", "kern.memorystatus_level"]).strip()
        if level.isdigit():
            out["mem_free_pct"] = int(level)  # macOS's own "memory free" level
        swap = parse_swap(self._try(["sysctl", "-n", "vm.swapusage"]))
        if swap is not None:
            out["swap_used_mb"] = swap
        try:
            out["load_1m"] = round(os.getloadavg()[0], 2)
        except OSError:
            pass
        return out
