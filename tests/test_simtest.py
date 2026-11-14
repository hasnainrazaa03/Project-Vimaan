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


# ---- Tier 2 checklist, end to end on the fake sim ---------------------------
from simtest import scenario, tier2  # noqa: E402

RADIO = "sim/cockpit2/radios/actuators/"
XPDR_MODE = RADIO + "transponder_mode"
XPDR_IDENT = "sim/cockpit2/radios/indicators/transponder_id"

# What a Tier 2 model makes of each checklist phrase (the gold set's shapes).
TIER2_NLU = {
    "autopilot on": ("toggle_autopilot", {"state": "on"}),
    "heading mode": ("set_autopilot_mode", {"mode": "heading"}),
    "altitude hold": ("set_autopilot_mode", {"mode": "altitude hold"}),
    "com one standby one two one decimal niner": (
        "set_com_standby",
        {"com_port": "1", "frequency": "121.9"},
    ),
    "flip com one": ("swap_com_frequency", {"com_port": "1"}),
    "com two standby one three two decimal four five": (
        "set_com_standby",
        {"com_port": "2", "frequency": "132.45"},
    ),
    "tune nav one to one one zero decimal five": (
        "set_nav_frequency",
        {"nav_port": "1", "nav_frequency": "110.5"},
    ),
    "course zero niner zero": ("set_course", {"degrees": "090"}),
    "nav two standby one one three decimal niner": (
        "set_nav_standby",
        {"nav_port": "2", "nav_frequency": "113.9"},
    ),
    "swap nav two": ("swap_nav_frequency", {"nav_port": "2"}),
    "nav two course one eight zero": ("set_course", {"degrees": "180", "nav_port": "2"}),
    "squawk four two one seven": ("set_transponder_code", {"squawk": "4217"}),
    "squawk zero one two zero": ("set_transponder_code", {"squawk": "0120"}),
    "squawk one two zero zero": ("set_transponder_code", {"squawk": "1200"}),
    "squawk seven seven zero zero": ("set_transponder_code", {"squawk": "7700"}),
    "transponder alt": ("set_transponder_mode", {"xpdr_mode": "alt"}),
    "squawk standby": ("set_transponder_mode", {"xpdr_mode": "standby"}),
    "ident": ("set_transponder_mode", {"xpdr_mode": "ident"}),
    "transponder on": ("set_transponder_mode", {"xpdr_mode": "on"}),
    "transponder off": ("set_transponder_mode", {"xpdr_mode": "off"}),
    "transponder altitude": ("set_transponder_mode", {"xpdr_mode": "alt"}),
}


def _generic_sim(xp):
    """The Baron: X-Plane's own autopilot, radios and transponder."""
    ap = "sim/cockpit2/autopilot/"
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.effects.update(
        {
            "sim/autopilot/servos_toggle": lambda x: x.datarefs.update({ap + "servos_on": 1}),
            "sim/autopilot/heading": lambda x: x.datarefs.update({ap + "heading_status": 2}),
            "sim/autopilot/altitude_hold": lambda x: x.datarefs.update(
                {ap + "altitude_hold_status": 2}
            ),
            "sim/transponder/transponder_ident": lambda x: x.datarefs.update({XPDR_IDENT: 1}),
        }
    )
    for mode, n in (("off", 0), ("standby", 1), ("on", 2), ("alt", 3)):
        xp.effects[f"sim/transponder/transponder_{mode}"] = lambda x, n=n: x.datarefs.update(
            {XPDR_MODE: n}
        )
    for radio, ports in (("com", "12"), ("nav", "12")):
        for port in ports:
            if radio == "com":
                active = f"{RADIO}com{port}_frequency_hz_833"
                standby = f"{RADIO}com{port}_standby_frequency_hz_833"
            else:
                active = f"{RADIO}nav{port}_frequency_hz"
                standby = f"{RADIO}nav{port}_standby_frequency_hz"
            xp.effects[f"sim/radios/{radio}{port}_standy_flip"] = lambda x, a=active, s=standby: (
                x.datarefs.update({a: x.datarefs[s], s: x.datarefs[a]})
            )


def _clicks(xp, position, down, up, low, high):
    def turn(sign):
        def press(x):
            x.datarefs[position] = max(low, min(high, x.datarefs[position] + sign))

        return press

    xp.effects[down], xp.effects[up] = turn(-1), turn(1)


def _b738(plugin, xp):
    import test_plugin

    test_plugin.aircraft(plugin, xp, commands=test_plugin.B738_XPDR)
    pos = "laminar/B738/knob/transponder_pos"
    xp.datarefs[pos], xp.datarefs[XPDR_MODE] = 1, 1  # STBY
    _clicks(
        xp,
        pos,
        "laminar/B738/knob/transponder_mode_dn",
        "laminar/B738/knob/transponder_mode_up",
        0,
        5,
    )
    xp.effects["laminar/B738/push_button/transponder_ident_dn"] = lambda x: x.datarefs.update(
        {XPDR_IDENT: 1}
    )

    def script(x):  # B738.systems: the knob handlers set X-Plane's mode
        p = round(x.datarefs[pos])
        x.datarefs[XPDR_MODE] = {0: 3, 1: 1}.get(p, 2)

    return script


def _a333(plugin, xp):
    import test_plugin

    k = "laminar/A333/transponder/"
    test_plugin.aircraft(
        plugin,
        xp,
        commands=test_plugin.A333_XPDR_COMMANDS,
        **{k + "alt_rpt_knob_pos": 0, k + "ta_ra_knob_pos": 0},
    )
    xp.datarefs[k + "auto_on_knob_pos"] = 0  # AUTO
    _clicks(xp, k + "auto_on_knob_pos", k + "auto_on_off_left", k + "auto_on_off_right", -1, 1)
    _clicks(xp, k + "alt_rpt_knob_pos", k + "alt_rpt_off", k + "alt_rpt_on", 0, 1)
    _clicks(xp, k + "ta_ra_knob_pos", k + "ta_ra_left", k + "ta_ra_right", 0, 2)
    xp.effects["sim/transponder/transponder_ident"] = lambda x: x.datarefs.update({XPDR_IDENT: 1})

    def script(x):  # A333.transponder, airborne, AC power on: every frame
        auto, alt, tcas = (
            round(x.datarefs[k + n])
            for n in ("auto_on_knob_pos", "alt_rpt_knob_pos", "ta_ra_knob_pos")
        )
        if auto == -1 or tcas == 0:
            x.datarefs[XPDR_MODE] = 1
        elif alt == 0:
            x.datarefs[XPDR_MODE] = 2
        else:
            x.datarefs[XPDR_MODE] = 6 if tcas == 1 else 7

    return script


def _run_checklist(monkeypatch, tmp_path, aircraft):
    import test_plugin

    test_plugin.NLU.clear()
    for text, (intent, slots) in TIER2_NLU.items():
        test_plugin.NLU[text] = (intent, slots, 0.99)
    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    results: list = []
    try:
        monkeypatch.setattr(module, "BRIDGE_POLL_SEC", 0.0)
        monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
        _generic_sim(xp)
        script = {"737": _b738, "a330": _a333}.get(aircraft, lambda p, x: None)(plugin, xp)
        if aircraft == "baron":
            steps = tier2.steps("baron")
        else:  # the transponder part: the selectors are what differs
            steps = tier2._transponder(aircraft)
        br = Bridge(tmp_path / "Vimaan_Logs", poll_sec=0.002)
        br.enable()
        plugin._bridge_next = 0.0
        runner = scenario.Runner(br, voice=None, poll_sec=0.005, log=lambda _line: None)
        t = threading.Thread(target=lambda: results.extend(runner.run(steps)), daemon=True)
        t.start()
        while t.is_alive():  # X-Plane's flight loop
            if script is not None:
                script(xp)
            plugin._flight_loop()
            t.join(0.001)
    finally:
        plugin.XPluginStop()
    return results


@pytest.mark.parametrize("aircraft", ["baron", "737", "a330"])
def test_the_tier2_checklist_passes_on_the_fake_sim(monkeypatch, tmp_path, aircraft):
    results = _run_checklist(monkeypatch, tmp_path, aircraft)
    failed = {r["step"]: r["problems"] for r in results if not r["ok"]}
    assert failed == {}
    assert not any(r.get("skipped") for r in results)
    names = [r["step"] for r in results]
    off = "transponder off" if aircraft == "baron" else "transponder off refused (no off)"
    assert off in names


def test_tier2_needs_a_tier2_model():
    assert cli._model_lacks("tier2", {"model_intents": ["None", "set_com_frequency"]})
    assert cli._model_lacks("tier2", {"model_intents": ["set_transponder_code"]}) is None
    assert cli._model_lacks("tier2", {}) is None  # an older plugin doesn't say
    assert cli._model_lacks("tier1", {"model_intents": ["None"]}) is None
