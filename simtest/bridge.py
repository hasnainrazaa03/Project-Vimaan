"""The runner's side of the test bridge: write a request, wait for its result."""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "ML") not in sys.path:
    sys.path.insert(0, str(_REPO / "ML"))

from vimaan_nlu import sim_bridge  # noqa: E402

# Long enough for a said command plus its read-backs (sim_bridge.SAY_TIMEOUT_SEC).
SAY_WAIT_SEC = sim_bridge.SAY_TIMEOUT_SEC + 5.0
WAIT_SEC = 10.0


class NoAnswer(RuntimeError):
    """The plugin didn't answer: X-Plane not running, Vimaan not loaded, or
    the bridge off."""


def default_log_dir() -> Path:
    """Where the plugin writes (ML/vimaan_nlu/telemetry.py)."""
    return Path(os.environ.get("VIMAAN_LOG_DIR") or Path.home() / "Vimaan_Logs")


class Bridge:
    def __init__(self, log_dir: str | os.PathLike[str] | None = None, poll_sec: float = 0.05):
        self.log_dir = Path(log_dir) if log_dir else default_log_dir()
        self.flag, self.inbox, self.outbox = sim_bridge.paths(self.log_dir)
        self.poll_sec = poll_sec

    # ---- the switch -------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.flag.exists()

    def enable(self) -> None:
        self.flag.parent.mkdir(parents=True, exist_ok=True)
        self.flag.write_text("Remove this file (or run: python -m simtest bridge off) to stop.\n")

    def disable(self) -> None:
        self.flag.unlink(missing_ok=True)

    # ---- requests ---------------------------------------------------------
    def request(self, op: str, timeout: float | None = None, **fields: Any) -> dict[str, Any]:
        """Send one request and wait for its result."""
        if not self.enabled:
            raise NoAnswer("the bridge is off: python -m simtest bridge on")
        rid = f"{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}"
        req = {"id": rid, "op": op, **fields}
        sim_bridge.write_json(self.inbox / f"{rid}.json", req)
        out = self.outbox / f"{rid}.json"
        wait = timeout if timeout is not None else (SAY_WAIT_SEC if op == "say" else WAIT_SEC)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if out.exists():
                result = sim_bridge.read_json(out)
                out.unlink(missing_ok=True)
                return result
            time.sleep(self.poll_sec)
        (self.inbox / f"{rid}.json").unlink(missing_ok=True)  # don't run it later
        raise NoAnswer(f"no answer to {op} in {wait:.0f} s: is X-Plane running with Vimaan loaded?")

    def state(self) -> dict[str, Any]:
        return self.request("state")

    def say_text(self, text: str) -> dict[str, Any]:
        return self.request("say", text=text)

    def say_wav(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        return self.request("say", wav=str(Path(path).resolve()))

    def read(self, *datarefs: str) -> dict[str, Any]:
        return self.request("read", datarefs=list(datarefs))["values"]

    def write(self, dataref: str, value: float | int) -> dict[str, Any]:
        return self.request("write", dataref=dataref, value=value)

    def command(self, name: str, phase: str = "once") -> dict[str, Any]:
        return self.request("command", name=name, phase=phase)

    def place(
        self,
        latitude: float,
        longitude: float,
        elevation_ft: float,
        heading_true: float,
        speed_kt: float,
    ) -> dict[str, Any]:
        return self.request(
            "place",
            latitude=latitude,
            longitude=longitude,
            elevation_ft=elevation_ft,
            heading_true=heading_true,
            speed_kt=speed_kt,
        )

    def airport(self, icao: str) -> dict[str, Any]:
        return self.request("airport", icao=icao.upper())

    def aircraft(self, path: str) -> dict[str, Any]:
        return self.request("aircraft", path=path)
