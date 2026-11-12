"""X-Plane itself, from the runner: start it through Steam, start a flight
through its web API, and quit. Learned on the first live run (2026-11-12):

- The Steam edition starts only through Steam (``steam://run/2014780``);
  opening X-Plane.app directly does nothing. Steam asks which launch option
  (X-Plane / Plane Maker / Airfoil Maker) unless a default is set.
- ``POST /api/v3/flight`` starts a flight (here: in the air, engines running),
  but X-Plane leaves its Flight Configuration screen open over it, paused.
  ``sim/operation/toggle_flight_config`` closes it.
- ``sim/operation/quit`` asks "Quit the simulator?" unless "Don't ask again"
  was ticked once.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API = "http://127.0.0.1:8086/api/v3"
STEAM_APP_ID = 2014780
FOLDER = Path.home() / "Library/Application Support/Steam/steamapps/common/X-Plane 12"
FEET = 0.3048
KNOT = 0.514444
# The four Tier 1 aircraft, relative to the X-Plane folder.
AIRCRAFT = {
    "baron": "Aircraft/Laminar Research/Beechcraft Baron 58/Baron_58.acf",
    "kingair": "Aircraft/Laminar Research/Beechcraft King Air C90B/C90B.acf",
    "737": "Aircraft/Laminar Research/Boeing 737-800/b738.acf",
    "a330": "Aircraft/Laminar Research/Airbus A330-300/A330.acf",
}


class XPlaneError(RuntimeError):
    pass


def _get(path: str) -> Any:
    with urllib.request.urlopen(f"{API}{path}", timeout=5) as r:
        return json.load(r)


def _post(path: str, body: dict[str, Any]) -> Any:
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode()
    return json.loads(text) if text.strip() else None


def running() -> bool:
    """Is X-Plane up with its web API answering?"""
    try:
        urllib.request.urlopen(API.rsplit("/", 1)[0] + "/capabilities", timeout=2).close()
        return True
    except (urllib.error.URLError, OSError):
        return False


def _id(kind: str, name: str) -> int:
    q = urllib.parse.quote("filter[name]") + "=" + urllib.parse.quote(name)
    found = _get(f"/{kind}?{q}").get("data") or []
    if not found:
        raise XPlaneError(f"X-Plane has no {kind[:-1]} {name}")
    return int(found[0]["id"])


def command(name: str) -> None:
    _post(f"/command/{_id('commands', name)}/activate", {"duration": 0})


def read(name: str) -> Any:
    return _get(f"/datarefs/{_id('datarefs', name)}/value")["data"]


def air_start_body(
    aircraft: str, lat: float, lon: float, alt_ft: float, hdg_true: float, kt: float
) -> dict[str, Any]:
    """The flight to ask X-Plane for: ``aircraft`` (a key of AIRCRAFT, or a
    path under the X-Plane folder) in the air, engines running."""
    return {
        "data": {
            "aircraft": {"path": AIRCRAFT.get(aircraft, aircraft)},
            "lle_air_start": {
                "latitude": lat,
                "longitude": lon,
                "elevation_in_meters": round(alt_ft * FEET, 1),
                "heading_true": hdg_true,
                "speed_in_meters_per_second": round(kt * KNOT, 2),
            },
            "engine_status": {"all_engines": {"running": True}},
        }
    }


def safe_mode(folder: Path = FOLDER) -> bool:
    """Did this X-Plane session start in Safe Mode (plugins, so Vimaan, off)?"""
    try:
        head = (folder / "Log.txt").read_text(errors="replace")[:20000]
    except OSError:
        return False
    return "SAFE MODE" in head


def launch(wait_sec: float = 300.0, poll_sec: float = 2.0) -> None:
    """Start X-Plane through Steam and wait for its web API. Steam may ask
    which launch option to use: someone has to pick "X-Plane" once."""
    if running():
        return
    subprocess.run(["open", f"steam://run/{STEAM_APP_ID}"], check=True)
    deadline = time.monotonic() + wait_sec
    while time.monotonic() < deadline:
        if running():
            time.sleep(poll_sec)  # the log's header is written by now
            if safe_mode():
                raise XPlaneError(
                    "X-Plane started in Safe Mode (plugins off, so no Vimaan): quit and launch again"
                )
            return
        time.sleep(poll_sec)
    raise XPlaneError("X-Plane didn't come up: is Steam waiting for a launch option to be picked?")


def start_flight(body: dict[str, Any], wait_sec: float = 120.0, poll_sec: float = 1.0) -> None:
    """Start a flight and get it running: close the Flight Configuration
    screen X-Plane leaves open (paused) over a flight started this way."""
    _post("/flight", body)
    want = body["data"].get("lle_air_start") or {}
    deadline = time.monotonic() + wait_sec
    while time.monotonic() < deadline:  # loaded: at the requested position
        time.sleep(poll_sec)
        try:
            lat = float(read("sim/flightmodel/position/latitude"))
        except (urllib.error.URLError, OSError, XPlaneError):
            continue
        if not want or abs(lat - float(want["latitude"])) < 0.01:
            break
    else:
        raise XPlaneError("the flight didn't load")
    for _ in range(3):
        if not read("sim/time/paused"):
            return
        command("sim/operation/toggle_flight_config")  # only while it's showing (paused)
        time.sleep(3)
    if read("sim/time/paused"):
        command("sim/operation/pause_off")
        time.sleep(1)
    if read("sim/time/paused"):
        raise XPlaneError("the flight is still paused")


def quit_xplane() -> None:
    """Ask X-Plane to quit (it may ask to confirm, unless told not to once)."""
    if running():
        command("sim/operation/quit")
