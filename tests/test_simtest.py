"""The test runner's side of the bridge (simtest/), against a stand-in
plugin, and end to end against the real plugin on the fake sim."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vimaan_nlu import sim_bridge  # noqa: E402

from simtest import __main__ as cli  # noqa: E402
from simtest import voice  # noqa: E402
from simtest.bridge import Bridge, NoAnswer  # noqa: E402


def responder(bridge, answer, stop):
    """Plays the plugin: answers each request with answer(req)."""
    while not stop.is_set():
        for req in sim_bridge.take_requests(bridge.inbox):
            sim_bridge.write_json(
                bridge.outbox / f"{req['id']}.json", {"id": req["id"], **answer(req)}
            )
        stop.wait(0.01)


@pytest.fixture
def br(tmp_path):
    b = Bridge(tmp_path, poll_sec=0.01)
    b.enable()
    stop = threading.Event()
    t = threading.Thread(
        target=responder, args=(b, lambda r: {"ok": True, "echo": r}, stop), daemon=True
    )
    t.start()
    yield b
    stop.set()
    t.join(1)


def test_requests_round_trip(br):
    got = br.request("read", datarefs=["sim/a"])
    assert got["ok"] and got["echo"]["op"] == "read" and got["echo"]["datarefs"] == ["sim/a"]
    assert br.place(47.5, -122.3, 6000, 180, 150)["echo"]["elevation_ft"] == 6000
    assert br.airport("ksea")["echo"]["icao"] == "KSEA"
    assert not list(br.outbox.glob("*.json"))  # results are taken


def test_no_answer_when_off_or_nobody_listens(tmp_path):
    b = Bridge(tmp_path, poll_sec=0.01)
    with pytest.raises(NoAnswer, match="bridge is off"):
        b.state()
    b.enable()
    with pytest.raises(NoAnswer, match="no answer"):
        b.request("state", timeout=0.1)
    assert not list(b.inbox.glob("*.json"))  # an unanswered request isn't left to run later


def test_the_cli_switch(tmp_path, capsys):
    assert cli.main(["--log-dir", str(tmp_path), "bridge", "on"]) == 0
    assert (tmp_path / "test" / "bridge.on").exists() and "Bridge on" in capsys.readouterr().out
    assert cli.main(["--log-dir", str(tmp_path), "bridge", "off"]) == 0
    assert not (tmp_path / "test" / "bridge.on").exists()
    assert cli.main(["--log-dir", str(tmp_path), "status"]) == 2  # off: refused, not hung


def test_the_say_line():
    line = cli._say_line(
        {
            "ok": True,
            "heard": "gear up",
            "intent": "toggle_landing_gear",
            "slots": {"state": "up"},
            "outcome": "executed",
            "replies": ["Gear up"],
            "latency_ms": 320,
            "checks": [{"ok": True}, {"ok": False}],
        }
    )
    assert "“gear up” → toggle_landing_gear state=up → executed" in line
    assert "said: Gear up" in line and "checks 1/2 ok" in line
    assert cli._say_line({"ok": False, "error": "busy"}) == "✕ busy"


@pytest.mark.skipif(not voice.available(), reason="needs macOS 'say'")
def test_generated_speech_is_16k_mono(tmp_path):
    import wave

    path = voice.synthesize("gear up", voice.DEFAULT_VOICE, cache_dir=tmp_path)
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
    assert voice.synthesize("gear up", voice.DEFAULT_VOICE, cache_dir=tmp_path) == path  # cached


def test_voice_names_parse(monkeypatch):
    out = (
        "Daniel (English (UK)) en_GB    # Hello! My name is Daniel.\n"
        "Fred                en_US    # Hello! My name is Fred.\n"
        "Thomas              fr_FR    # Bonjour, je m’appelle Thomas.\n"
    )
    monkeypatch.setattr(voice, "available", lambda: True)
    monkeypatch.setattr(voice.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": out})())
    assert voice.voices() == ["Daniel", "Fred"]


# ---- end to end: runner → files → the real plugin on the fake sim ---------
def test_the_runner_drives_the_plugin(monkeypatch, tmp_path):
    import test_plugin

    test_plugin.NLU.clear()
    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    try:
        monkeypatch.setattr(module, "BRIDGE_POLL_SEC", 0.0)
        monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
        br = Bridge(tmp_path / "Vimaan_Logs", poll_sec=0.005)
        br.enable()
        plugin._bridge_next = 0.0
        test_plugin.NLU["autopilot on"] = ("toggle_autopilot", {"state": "on"}, 0.99)
        xp.effects["sim/autopilot/servos_toggle"] = lambda x: x.datarefs.update(
            {"sim/cockpit2/autopilot/servos_on": 1}
        )
        results = {}
        runner = threading.Thread(
            target=lambda: results.update(
                state=br.state(),
                said=br.say_text("autopilot on"),
                read=br.read("sim/cockpit2/autopilot/servos_on"),
            ),
            daemon=True,
        )
        runner.start()
        while runner.is_alive():  # X-Plane's flight loop
            plugin._flight_loop()
            runner.join(0.002)
    finally:
        plugin.XPluginStop()
    assert results["state"]["ready"] and results["state"]["aircraft"]["icao"] == "BE58"
    said = results["said"]
    assert said["outcome"] == "executed" and said["replies"] == ["Autopilot 1 on"]
    assert results["read"] == {"sim/cockpit2/autopilot/servos_on": 1}
    assert json.dumps(said)  # plain JSON all the way


# ---- X-Plane itself (simtest/xplane.py), with the web API faked ----------
from simtest import xplane  # noqa: E402


def test_air_start_body():
    body = xplane.air_start_body("baron", 47.45, -122.31, 6000, 180, 150)
    d = body["data"]
    assert d["aircraft"]["path"].endswith("Beechcraft Baron 58/Baron_58.acf")
    assert d["lle_air_start"] == {
        "latitude": 47.45,
        "longitude": -122.31,
        "elevation_in_meters": 1828.8,
        "heading_true": 180,
        "speed_in_meters_per_second": 77.17,
    }
    assert d["engine_status"] == {"all_engines": {"running": True}}
    assert xplane.air_start_body("Aircraft/x/y.acf", 0, 0, 0, 0, 0)["data"]["aircraft"]["path"] == (
        "Aircraft/x/y.acf"
    )


class FakeAPI:
    """X-Plane's web API: a flight that loads, paused under its Flight
    Configuration screen until that's toggled."""

    def __init__(self):
        self.state = {"sim/flightmodel/position/latitude": 0.0, "sim/time/paused": 1}
        self.commands = []

    def get(self, path):
        if path.startswith(("/datarefs?", "/commands?")):
            name = xplane.urllib.parse.unquote(path.split("=", 1)[1])
            return {"data": [{"id": hash(name) & 0xFFFF, "name": name}]}
        name = next(n for n in self.state if str(hash(n) & 0xFFFF) in path)
        return {"data": self.state[name]}

    def post(self, path, body):
        if path == "/flight":
            self.state["sim/flightmodel/position/latitude"] = body["data"]["lle_air_start"][
                "latitude"
            ]
        else:
            cid = int(path.split("/")[2])
            name = next(
                n
                for n in (
                    "sim/operation/toggle_flight_config",
                    "sim/operation/pause_off",
                    "sim/operation/quit",
                )
                if hash(n) & 0xFFFF == cid
            )
            self.commands.append(name)
            if name == "sim/operation/toggle_flight_config":
                self.state["sim/time/paused"] = 0


def test_a_started_flight_is_brought_out_from_under_the_config_screen(monkeypatch):
    api = FakeAPI()
    monkeypatch.setattr(xplane, "_get", api.get)
    monkeypatch.setattr(xplane, "_post", api.post)
    monkeypatch.setattr(xplane.time, "sleep", lambda s: None)
    xplane.start_flight(xplane.air_start_body("baron", 47.45, -122.31, 6000, 180, 150))
    assert api.commands == ["sim/operation/toggle_flight_config"]
    assert api.state["sim/time/paused"] == 0


def test_safe_mode_is_spotted(tmp_path):
    (tmp_path / "Log.txt").write_text("...\n0:00:08.769 I/SYS: **   SAFE MODE   **\n")
    assert xplane.safe_mode(tmp_path)
    (tmp_path / "Log.txt").write_text("Log.txt for X-Plane 12.4.3\nFetching plugins\n")
    assert not xplane.safe_mode(tmp_path)
    assert not xplane.safe_mode(tmp_path / "missing")


def test_flap_limits_come_from_the_aircraft_file(tmp_path):
    from simtest.aircraft_file import flap_limits

    acf = tmp_path / "b738.acf"
    acf.write_bytes(
        b"I\n1200 version\nP acf/_Vfe1_kts 250.000000000\nP acf/_Vfem_kts 162.000000000\n"
    )
    assert flap_limits(acf) == {"first": 250.0, "full": 162.0}
    assert flap_limits(tmp_path / "missing.acf") == {}
