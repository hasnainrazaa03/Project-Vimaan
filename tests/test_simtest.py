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
        self.state = {
            "sim/flightmodel/position/latitude": 0.0,
            "sim/flightmodel/position/longitude": 0.0,
            "sim/time/paused": 1,
        }
        self.commands = []
        self.flights = []

    def get(self, path):
        if path.startswith(("/datarefs?", "/commands?")):
            name = xplane.urllib.parse.unquote(path.split("=", 1)[1])
            return {"data": [{"id": hash(name) & 0xFFFF, "name": name}]}
        name = next(n for n in self.state if str(hash(n) & 0xFFFF) in path)
        return {"data": self.state[name]}

    def post(self, path, body):
        if path == "/flight":
            self.flights.append(body)
            air = body["data"].get("lle_air_start")
            if air:
                lat, lon = air["latitude"], air["longitude"]
            else:  # on a 5 nm final, west of the airport
                lat, lon = xplane.AIRPORTS[body["data"]["runway_start"]["airport_id"]]
                lon -= 0.1
            self.state["sim/flightmodel/position/latitude"] = lat
            self.state["sim/flightmodel/position/longitude"] = lon
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


def test_runway_start_body():
    body = xplane.runway_start_body("737", "KPDX", "10R")
    d = body["data"]
    assert d["aircraft"]["path"].endswith("Boeing 737-800/b738.acf")
    assert d["runway_start"] == {"airport_id": "KPDX", "runway": "10R"}
    assert d["engine_status"] == {"all_engines": {"running": True}}
    final = xplane.runway_start_body("baron", "KPDX", "10R", final_nm=5)["data"]["runway_start"]
    assert final["final_distance_in_nautical_miles"] == 5


def test_a_runway_start_is_loaded_once_the_aircraft_is_near_its_airport(monkeypatch):
    api = FakeAPI()
    monkeypatch.setattr(xplane, "_get", api.get)
    monkeypatch.setattr(xplane, "_post", api.post)
    monkeypatch.setattr(xplane.time, "sleep", lambda s: None)
    xplane.start_flight(xplane.runway_start_body("baron", "KPDX", "10R", final_nm=5))
    assert api.state["sim/time/paused"] == 0  # on a 5 nm final, brought out from the screen
    lat, lon, tol = xplane._where(api.flights[0])
    assert (lat, lon) == xplane.AIRPORTS["KPDX"]
    # The tier flights' area west of Portland isn't near it: no early "loaded".
    assert abs(xplane.TEST_AREA[1] - lon) > tol


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


# --- Tier 3 on the fake sim ---------------------------------------------------

from simtest import tier3  # noqa: E402

SW3, ICE3 = "sim/cockpit2/switches/", "sim/cockpit2/ice/"


def _light(light, switch):
    return ("toggle_lights", {"light": light, "switch": switch})


TIER3_NLU = {
    "beacon on": _light("beacon", "on"),
    "landing lights on": _light("landing", "on"),
    "strobes on": _light("strobe", "on"),
    "nav lights on": _light("nav", "on"),
    "taxi lights off": _light("taxi", "off"),
    "strobes auto": _light("strobe", "auto"),
    "logo lights on": _light("logo", "on"),
    "seat belt sign on": ("toggle_signs", {"sign": "seat belt", "switch": "on"}),
    "seat belt sign auto": ("toggle_signs", {"sign": "seat belt", "switch": "auto"}),
    "no smoking sign on": ("toggle_signs", {"sign": "no smoking", "switch": "on"}),
    "no smoking sign auto": ("toggle_signs", {"sign": "no smoking", "switch": "auto"}),
    "yaw damper on": ("toggle_yaw_damper", {"state": "on"}),
    **{
        f"{word} {setting}": ("toggle_anti_ice", {"ice": ice, "switch": setting})
        for word, ice in (
            ("pitot heat", "pitot"),
            ("probe heat", "pitot"),
            ("window heat", "window"),
            ("wing anti ice", "wing"),
            ("boots", "wing"),
            ("engine anti ice", "engine"),
            ("prop heat", "prop"),
            ("propeller heat", "prop"),
        )
        for setting in ("on", "off", "auto")
    },
    "avionics on": ("toggle_electrical", {"power": "avionics", "switch": "on"}),
    "generator two off": (
        "toggle_electrical",
        {"power": "generator", "instance": "2", "switch": "off"},
    ),
    "fuel pumps on": ("toggle_fuel_pump", {"switch": "on"}),
    "fuel pumps off": ("toggle_fuel_pump", {"switch": "off"}),
    "speed brake armed": ("set_speed_brake", {"speedbrake": "armed"}),
    "stow the speed brakes": ("set_speed_brake", {"speedbrake": "retract"}),
    "auto brake two": ("set_autobrake", {"autobrake_level": "2"}),
    "auto brake medium": ("set_autobrake", {"autobrake": "medium"}),
    "auto brake off": ("set_autobrake", {"autobrake": "off"}),
}


def _baron3(plugin, xp):
    import test_plugin

    test_plugin.aircraft(plugin, xp, engine_type=1, commands=test_plugin.B58_COMMANDS)
    xp.datarefs["sim/aircraft/parts/acf_sbrkEQ"] = 0  # its .acf: no speed brakes
    # Its replaced wing_boot commands: "on" is 2 (single, auto-deflating).
    xp.effects["sim/ice/wing_boot_on"] = lambda x: x.datarefs.update(
        {ICE3 + "ice_surface_boot_on": 2}
    )


def _b738_3(plugin, xp):
    import test_plugin
    from vimaan_nlu import switches

    test_plugin.aircraft(plugin, xp, engine_type=7, commands=test_plugin.B738_SWITCHES)
    for knob, low, high in (
        (switches.B738_POSITION, -1, 1),
        (switches.B738_SEATBELT, 0, 2),
        (switches.B738_TAXI, 0, 2),
    ):
        xp.datarefs[knob.position] = 0
        _clicks(xp, knob.position, knob.down, knob.up, low, high)
    xp.datarefs[switches.B738_SPEEDBRAKE] = 0.0
    xp.datarefs[SW3 + "taxi_light_on"] = 0
    pos = switches.B738_POSITION.position

    def script(x):  # B738.lighting: the position switch fires X-Plane's lights
        x.datarefs[SW3 + "strobe_lights_on"] = int(round(x.datarefs[pos]) == 1)
        x.datarefs[SW3 + "navigation_lights_on"] = int(round(x.datarefs[pos]) != 0)

    return script


def _a333_3(plugin, xp):
    import test_plugin
    from vimaan_nlu import switches

    test_plugin.aircraft(plugin, xp, engine_type=7, commands=test_plugin.A333_SWITCHES)
    for knob in (switches.A333_NAV, switches.A333_SEATBELT, switches.A333_NO_SMOKING):
        xp.datarefs[knob.position] = 0
        _clicks(xp, knob.position, knob.down, knob.up, 0, 2)
    for setting, value in (("on", 2), ("auto", 1), ("off", 0)):
        xp.effects[f"laminar/A333/toggle_switch/strobe_pos_{setting}"] = lambda x, v=value: (
            x.datarefs.update({switches.A333_STROBE: v})
        )
    xp.effects["laminar/A333/switch/lighting/taxi_light_off"] = lambda x: x.datarefs[
        SW3 + "landing_lights_switch"
    ].__setitem__(1, 0.0)
    toggles = {
        "laminar/A333/monitors/probe_window_heat": "laminar/A333/buttons/probe_window_heat_toggle",
        "laminar/A333/buttons/wing_anti_ice": "sim/ice/wing_heat_tog",
        "laminar/A333/buttons/engine_anti_ice1": "sim/ice/inlet_heat0_tog",
        "laminar/A333/buttons/engine_anti_ice2": "sim/ice/inlet_heat1_tog",
    }
    for state, command in toggles.items():
        xp.datarefs[state] = 0
        xp.custom_commands.setdefault(command, None)
        xp.effects[command] = lambda x, s=state: x.datarefs.update({s: 1 - x.datarefs[s]})

    def autobrake(level):  # LO / MED / MAX toggle: the lit one turns it off
        def press(x):
            now = x.datarefs[SW3 + "auto_brake_level"]
            x.datarefs[SW3 + "auto_brake_level"] = 1 if now == level else level

        return press

    xp.effects["sim/flight_controls/brakes_1_auto"] = autobrake(3)
    xp.effects["sim/flight_controls/brakes_2_auto"] = autobrake(4)


def _run_tier3(monkeypatch, tmp_path, aircraft, steps):
    import test_plugin

    test_plugin.NLU.clear()
    for text, (intent, slots) in {**TIER2_NLU, **TIER3_NLU}.items():
        test_plugin.NLU[text] = (intent, slots, 0.99)
    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    results: list = []
    try:
        monkeypatch.setattr(module, "BRIDGE_POLL_SEC", 0.0)
        monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
        _generic_sim(xp)
        xp.datarefs["sim/flightmodel/position/y_agl"] = 5000 * 0.3048
        script = {"baron": _baron3, "737": _b738_3, "a330": _a333_3}[aircraft](plugin, xp)
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


def test_the_tier3_checklist_passes_on_the_fake_baron(monkeypatch, tmp_path):
    results = _run_tier3(monkeypatch, tmp_path, "baron", tier3.steps("baron"))
    failed = {r["step"]: r["problems"] for r in results if not r["ok"]}
    assert failed == {}
    assert not any(r.get("skipped") for r in results)
    names = [r["step"] for r in results]
    assert {"no seat belt sign", "no speed brakes", "generator off is held in flight"} <= set(names)


@pytest.mark.parametrize("aircraft", ["737", "a330"])
def test_the_tier3_switches_that_differ_pass_on_the_fake_jets(monkeypatch, tmp_path, aircraft):
    steps = tier3._lights(aircraft) + tier3._signs(aircraft) + tier3._jets(aircraft)
    if aircraft == "a330":
        steps += [s for s in tier3._anti_ice("a330") if "prop" not in s.name]
    results = _run_tier3(monkeypatch, tmp_path, aircraft, steps)
    failed = {r["step"]: r["problems"] for r in results if not r["ok"]}
    assert failed == {}


def test_tier3_needs_a_tier3_model():
    assert cli._model_lacks("tier3", {"model_intents": ["set_transponder_code"]})
    assert cli._model_lacks("tier3", {"model_intents": ["set_autobrake"]}) is None


# --- Tier 4: questions on the question key, answers against the sim -----------

from simtest import tier4  # noqa: E402

# The question model's reading of each checklist question (as q1 reads them).
TIER4_QNLU = {
    "what's our altitude": ("ask_flight_data", {"reading": "altitude"}),
    "what's our airspeed": ("ask_flight_data", {"reading": "airspeed"}),
    "what's the heading": ("ask_flight_data", {"reading": "heading"}),
    "what's our vertical speed": ("ask_flight_data", {"reading": "vertical speed"}),
    "what's our ground speed": ("ask_flight_data", {"reading": "ground speed"}),
    "what's the wind": ("ask_flight_data", {"reading": "wind"}),
    "what's the outside temperature": ("ask_flight_data", {"reading": "temperature"}),
    "what's the altimeter setting": ("ask_flight_data", {"reading": "altimeter"}),
    "how high are we above the ground": ("ask_flight_data", {"reading": "height above ground"}),
    "what phase of flight are we in": ("ask_flight_data", {"reading": "phase"}),
    "what's our mach number": ("ask_flight_data", {"reading": "mach"}),
    "how much fuel do we have": ("ask_fuel", {}),
    "how long will our fuel last": ("ask_fuel_time", {}),
    "what's the oil pressure": ("ask_engine", {"engine_reading": "oil pressure"}),
    "what's the manifold pressure": ("ask_engine", {"engine_reading": "manifold pressure"}),
    "what's the torque": ("ask_engine", {"engine_reading": "torque"}),
    "what's the e g t": ("ask_engine", {"engine_reading": "e g t"}),
    "what's the n one": ("ask_engine", {"engine_reading": "n one"}),
    "what's com one": ("ask_com", {"com_port": "1"}),
    "what's com one standby": ("ask_com", {"com_port": "1", "which": "standby"}),
    "what's nav one": ("ask_nav", {"nav_port": "1"}),
    "what's our squawk": ("ask_transponder", {}),
    "autopilot status": ("ask_autopilot", {}),
    "what altitude is set": ("ask_autopilot", {"ap_target": "altitude"}),
    "what's the heading bug": ("ask_autopilot", {"ap_target": "heading"}),
    "are the landing lights on": ("ask_switch", {"light": "landing"}),
    "landing lights status": ("ask_switch", {"light": "landing"}),
    "is the gear down": ("ask_switch", {"switch_system": "gear"}),
    "where are the flaps": ("ask_switch", {"switch_system": "flaps"}),
    "how far to the next waypoint": ("ask_waypoint", {}),
    "what time is it": ("ask_time", {}),
    "status report": ("ask_status_generic", {}),
}
# What the command model makes of the ones it reads as commands.
TIER4_NLU = {
    "gear down": ("toggle_landing_gear", {"state": "down"}),
    "autopilot off": ("toggle_autopilot", {"state": "off"}),
    "landing lights status": ("toggle_lights", {"light": "landing"}),  # no on / off
}
G4 = "sim/cockpit2/gauges/indicators/"
E4 = "sim/cockpit2/engine/indicators/"


def _cruise(xp):
    """A twin in cruise, as an air start leaves it."""
    xp.datarefs.update(
        {
            G4 + "altitude_ft_pilot": 6004.0,
            G4 + "airspeed_kts_pilot": 151.4,
            G4 + "heading_AHARS_deg_mag_pilot": 269.6,
            G4 + "vvi_fpm_pilot": 40.0,
            G4 + "ground_speed_kt": 163.2,
            G4 + "mach_pilot": 0.412,
            G4 + "radio_altimeter_height_ft_pilot": 4972.0,
            G4 + "wind_heading_deg_mag": 45.0,
            G4 + "wind_speed_kts": 12.0,
            "sim/cockpit2/temperature/outside_air_temp_degc": -4.6,
            "sim/cockpit2/gauges/actuators/barometer_setting_in_hg_pilot": 29.92,
            "sim/flightmodel/weight/m_fuel_total": 10439.0,
            "sim/cockpit2/autopilot/heading_dial_deg_mag_pilot": 270.0,
            "sim/cockpit2/autopilot/altitude_dial_ft": 6000.0,
            "sim/cockpit2/controls/gear_handle_down": 0,
            "sim/time/zulu_time_sec": 52230.0,
        }
    )
    xp.datarefs["sim/flightmodel2/gear/deploy_ratio"] = [0.0] * 10
    for name, values in (
        ("fuel_flow_kg_sec", (0.0125, 0.0125)),
        ("oil_pressure_psi", (60.2, 61.4)),  # differ: said engine by engine
        ("MPR_in_hg", (23.5, 23.5)),
        ("torque_n_mtr", (1900.0, 1900.0)),
        ("EGT_deg_C", (566.0, 566.0)),
        ("N1_percent", (85.2, 85.4)),
    ):
        xp.datarefs[E4 + name][:2] = list(values)


def _run_tier4(monkeypatch, tmp_path, aircraft, steps, qnlu=None, datarefs=None):
    import test_plugin

    test_plugin.NLU.clear()
    for text, (intent, slots) in {**TIER2_NLU, **TIER3_NLU, **TIER4_NLU}.items():
        test_plugin.NLU[text] = (intent, slots, 0.99)
    readings = {**TIER4_QNLU, **(qnlu or {})}

    def question_model(text):  # q1 beside the command model, as the worker loads it
        intent, slots = readings.get(text, ("None", {}))
        return {"intent": intent, "slots": dict(slots), "confidence": 0.99, "normalized_text": text}

    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    results: list = []
    try:
        monkeypatch.setattr(module, "BRIDGE_POLL_SEC", 0.0)
        monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
        plugin.worker.worker.predict_question = question_model
        plugin._question_intents = {intent for intent, _ in TIER4_QNLU.values()} | {"None"}
        _generic_sim(xp)
        _cruise(xp)
        xp.datarefs["sim/flightmodel/position/y_agl"] = 5000 * 0.3048
        script = {"baron": _baron3, "737": _b738_3, "a330": _a333_3}[aircraft](plugin, xp)
        xp.datarefs.update(datarefs or {})
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
        state = plugin._bridge_state()
    finally:
        plugin.XPluginStop()
    return results, state, xp


def test_the_tier4_checklist_passes_on_the_fake_baron(monkeypatch, tmp_path):
    results, state, xp = _run_tier4(monkeypatch, tmp_path, "baron", tier4.steps("baron"))
    failed = {r["step"]: r["problems"] for r in results if not r["ok"]}
    assert failed == {}
    assert not any(r.get("skipped") for r in results)
    by_step = {r["step"]: r for r in results}
    assert by_step["fuel"]["replies"] == ["Fuel 3836 gallons"]  # avgas 6 lb/gal, from 10,439 kg
    assert by_step["oil pressure"]["replies"] == [
        "Engine 1 oil pressure 60 P S I, engine 2 oil pressure 61 P S I"
    ]
    assert by_step["altimeter"]["replies"] == ["Altimeter two niner niner two"]
    assert by_step["gear down on the question key is refused"]["replies"] == ["I can't answer that"]
    questions = [r for r in results if r["step"] in {s.name for s in tier4.steps("baron") if s.ask}]
    assert questions and all(r["actions"] == [] for r in questions)
    assert tier4.NEEDS_QUESTION_INTENT in state["question_intents"]
    assert xp.datarefs["sim/cockpit2/autopilot/servos_on"] == 1


@pytest.mark.parametrize("aircraft", ["737", "a330"])
def test_the_tier4_answers_that_differ_pass_on_the_fake_jets(monkeypatch, tmp_path, aircraft):
    names = {"altimeter", "fuel", "mach", "E G T", "N1", "transponder", "landing lights"}
    steps = [tier4._landing_lights_on(aircraft)]
    steps += [s for s in tier4.steps(aircraft) if s.name in names]
    results, _, _ = _run_tier4(monkeypatch, tmp_path, aircraft, steps)
    failed = {r["step"]: r["problems"] for r in results if not r["ok"]}
    assert failed == {}
    by_step = {r["step"]: r for r in results}
    unit = {"737": "Fuel 23000 pounds", "a330": "Fuel 10400 kilograms"}[aircraft]
    assert by_step["fuel"]["replies"] == [unit]
    assert by_step["mach"]["replies"] == ["Mach point four one"]


def test_the_altimeter_check_follows_the_737s_hpa_selector(monkeypatch, tmp_path):
    steps = [s for s in tier4.steps("737") if s.name == "altimeter"]
    selector = "laminar/B738/EFIS_control/capt/baro_in_hpa"
    results, _, _ = _run_tier4(monkeypatch, tmp_path, "737", steps, datarefs={selector: 1.0})
    (r,) = results
    assert r["ok"] and r["replies"] == ["Q N H one zero one three"]


def test_the_n1_step_catches_an_engine_reading_the_plugin_doesnt_know(monkeypatch, tmp_path):
    # Before the postprocessor read "n 1" back as "n one", q1's "what's the
    # n one" reached the plugin as engine_reading "n 1": the answer was "I
    # can't read that now", and the step catches an answer like it.
    steps = [s for s in tier4.steps("737") if s.name == "N1"]
    results, _, _ = _run_tier4(
        monkeypatch,
        tmp_path,
        "737",
        steps,
        qnlu={"what's the n one": ("ask_engine", {"engine_reading": "n 1"})},
    )
    (r,) = results
    assert not r["ok"] and r["replies"] == ["I can't read that now"]


def test_every_question_family_is_asked_on_every_aircraft():
    from simtest.tier1 import SETUPS

    for aircraft in FLEET:
        steps = tier4.steps(aircraft)
        names = {s.name for s in steps}
        for family, asked in tier4.FAMILIES.items():
            assert names & set(asked), (aircraft, family)
        assert ("mach" in names) == SETUPS[aircraft].jet
        for s in steps:
            if s.outcomes == ("answered",):
                assert s.no_actions and (s.answer is not None or s.reply), s.name
        assert sum(s.ask for s in steps) >= 25
        assert any(s.outcomes == ("answered",) and not s.ask for s in steps)  # the talk key too


@pytest.mark.parametrize(
    ("reply", "readings", "ok"),
    [
        ("Altitude 7400 feet", {tier4.ALT: 7404.0}, True),
        ("Altitude 7400 feet", {tier4.ALT: 7600.0}, False),
        ("Heading three six zero", {tier4.HDG: 1.5}, True),  # 0 is said as 360
        ("Heading two seven zero", {tier4.HDG: 90.0}, False),
    ],
)
def test_an_answer_is_checked_against_the_reading(reply, readings, ok):
    check = tier4.altitude_answer() if reply.startswith("Altitude") else tier4.heading_answer()
    assert check.ok(reply, readings) is ok


def test_answers_in_the_wrong_unit_or_mode_fail():
    pounds = tier4.fuel_answer(tier4.COCKPITS["737"])
    assert pounds.ok("Fuel 23000 pounds", {tier4.FUEL: 10439.0})
    assert not pounds.ok("Fuel 10400 kilograms", {tier4.FUEL: 10439.0})
    alt = tier4.altimeter_answer(tier4.COCKPITS["a330"])
    inches = {
        tier4.BARO_IN: 29.92,
        tier4.BARO_STD: 0,
        "laminar/A333/barometer/capt_inHg_hPa_pos": 0,
    }
    assert alt.ok("Altimeter two niner niner two", inches)
    assert not alt.ok("Q N H one zero one three", inches)
    hpa = {**inches, "laminar/A333/barometer/capt_inHg_hPa_pos": 1}
    assert alt.ok("Q N H one zero one three", hpa)
    assert alt.ok("Altimeter standard", {**inches, tier4.BARO_STD: 1})
    vs = tier4.vertical_speed_answer()
    assert vs.ok("Level", {tier4.VVI: 60.0}) and vs.ok(
        "Climbing 500 feet per minute", {tier4.VVI: 480.0}
    )
    assert not vs.ok("Descending 500 feet per minute", {tier4.VVI: 480.0})


def test_spoken_numbers():
    assert tier4.spoken_number("two niner niner two") == 2992
    assert tier4.spoken_number("one two one decimal niner") == 121.9
    assert tier4.spoken_number("two seven heading") is None


def test_tier4_needs_a_question_model():
    assert cli._model_lacks("tier4", {"model_intents": ["set_autobrake"], "question_intents": []})
    assert (
        cli._model_lacks("tier4", {}) is not None
    )  # an older plugin doesn't say: no question model
    assert cli._model_lacks("tier4", {"question_intents": ["None", "ask_flight_data"]}) is None


def test_a_question_goes_over_the_bridge_on_the_question_key(br):
    assert br.say_text("what's our altitude", ask=True)["echo"]["ask"] is True
    assert "ask" not in br.say_text("gear down")["echo"]  # the talk key, as before
    assert sim_bridge.problem({"id": "a", "op": "say", "text": "x", "ask": "yes"})
    assert sim_bridge.problem({"id": "a", "op": "say", "text": "x", "ask": True}) is None


class _MovingSim:
    """A bridge whose reading moves between reads: one read before the
    question, one after the answer."""

    def __init__(self, *egt):
        self.reads = iter(egt)

    def read(self, *refs):
        value = next(self.reads)
        return {ref: value for ref in refs}

    def say_text(self, text, ask=False):
        return {"ok": True, "outcome": "answered", "replies": ["E G T 620 degrees on both engines"]}


@pytest.mark.parametrize(("before", "after", "ok"), [(619.0, 608.3, True), (640.0, 608.3, False)])
def test_a_moving_reading_is_checked_just_before_the_question_too(before, after, ok):
    # In-sim 2026-11-18: the 737 answered "E G T 620" and the sim read 608.3
    # just after (the autothrottle hunting); the plugin read it in between.
    step = next(s for s in tier4.steps("737") if s.name == "E G T")
    runner = scenario.Runner(_MovingSim(before, after), voice=None, log=lambda _line: None)
    r = runner.run_step(step)
    assert r["ok"] is ok
    assert r["answer"]["readings_before"][tier4.ENG + "EGT_deg_C[0]"] == before
    if not ok:
        assert "just before the question: E G T 640.0 / 640.0 degrees" in r["problems"][0]


class _FlyingSim:
    """A bridge on a flight that moves on with each read: Vimaan's phase in
    its state, a dataref alongside."""

    def __init__(self, phases, ias):
        self.phases, self.ias = list(phases), list(ias)
        self.writes = []

    def state(self):
        return {"phase": self.phases.pop(0) if len(self.phases) > 1 else self.phases[0]}

    def read(self, *refs):
        assert scenario.PHASE not in refs  # not a dataref: never asked of the sim
        value = self.ias.pop(0) if len(self.ias) > 1 else self.ias[0]
        return dict.fromkeys(refs, value)

    def write(self, ref, value):
        self.writes.append((ref, value))
        return {"ok": True}


def test_a_step_waits_for_the_phase_vimaan_shows():
    sim = _FlyingSim(["parked", "parked", "taxi", "taxi", "takeoff_roll"], [0.0, 30.0, 45.0])
    runner = scenario.Runner(sim, voice=None, poll_sec=0.0, log=lambda _line: None)
    ias = "sim/cockpit2/gauges/indicators/airspeed_kts_pilot"
    results = runner.run(
        [
            scenario.Step(
                "throttle up",
                write=("sim/cockpit2/engine/actuators/throttle_ratio_all", 1.0),
                sim=(scenario.above("40 kt", ias, 40, within_sec=5),),
            ),
            scenario.Step(
                "takeoff roll",
                sim=(scenario.phase_is("takeoff_roll", within_sec=5),),
            ),
            scenario.Step("not landing", sim=(scenario.phase_is("landing", "rollout"),)),
        ]
    )
    assert sim.writes == [("sim/cockpit2/engine/actuators/throttle_ratio_all", 1.0)]
    assert [r["ok"] for r in results] == [True, True, False]
    assert results[1]["sim"][0]["actual"] == "takeoff_roll"
    assert results[2]["problems"] == [
        "phase landing or rollout: got takeoff_roll, wanted landing or rollout"
    ]
    assert scenario.below("slow", ias, 5).ok(4.0) and not scenario.below("slow", ias, 5).ok(None)


# ---- whole flights (simtest/flights.py, Phase F.3) ---------------------------
from simtest import flights  # noqa: E402


def _phases_checked(steps):
    return [c.expected for s in steps for c in s.sim if c.datarefs == (scenario.PHASE,)]


@pytest.mark.parametrize("aircraft", list(flights.PLANES))
def test_each_flight_checks_its_phases_in_order(aircraft):
    hands = flights.new_hands(aircraft)
    takeoff = _phases_checked(flights.takeoff(aircraft, hands))
    assert takeoff == [
        "parked",
        "taxi or takeoff_roll",
        "takeoff_roll",
        "initial_climb",
        "climb",
        "cruise",
    ]
    landing = _phases_checked(flights.landing(aircraft, hands))
    assert landing == ["approach", "landing", "rollout", "taxi or parked", "parked"]
    go_around = _phases_checked(flights.go_around(aircraft, hands))
    assert go_around == ["approach", "go_around", "climb", "cruise"]


@pytest.mark.parametrize("aircraft", list(flights.PLANES))
def test_jets_use_their_autothrottle_and_pistons_the_runners_throttle(aircraft):
    # An autothrottle flies the speed (737, A330, MD-82); otherwise the
    # runner's throttle hold (the pistons, the Citation X and SF50).
    jet = flights.PLANES[aircraft].autothrottle
    steps = flights.landing(aircraft, flights.new_hands(aircraft))
    said = [s.say for s in steps if s.say]
    held = [s.pilot.__name__ for s in steps if s.pilot is not None]
    assert ("auto throttle on" in said) is jet
    assert ("auto throttle off" in said) is jet  # before touchdown: thrust off
    assert any(h.startswith("speed hold") for h in held) is not jet
    assert "idle, brakes and rudder" in held
    # Thrust off from 50 ft on every aircraft: a speed held into the flare
    # floats it along the runway (the 737 at 145 kt, the Baron 30 s at 3 ft).
    names = [s.name for s in steps]
    assert names.index("50 ft") < names.index("thrust idle (pilot)") < names.index("touchdown")
    assert ("idle" in held) is not jet  # the pistons: the throttle hold stopped
    # Every phrase is one Vimaan reads back (no numbers it can't say).
    assert all(s.say != "speed " for s in steps)


def test_every_aircraft_flies_every_flight():
    # 2026-11-19: igniters (King Air), the APU first (jets), FADEC (A330).
    # Phase L.2: the three new jets' engine starts aren't profiled yet.
    for flight in flights.FLIGHTS:
        for a in flights.PLANES:
            blocked = flight == "restart" and a in ("citx", "md82", "sf50")
            assert (flights.cannot_fly(flight, a) != "") is blocked, (flight, a)


def test_the_restart_shuts_down_starts_and_takes_off():
    steps = flights.restart("baron", flights.new_hands("baron"))
    said = [s.say for s in steps if s.say]
    assert said[:2] == ["shut down both engines", "start both engines"]
    assert _phases_checked(steps)[-1] == "cruise"  # then the whole takeoff
    assert flights.start_body("restart", "baron")["data"]["runway_start"] == {
        "airport_id": "KPDX",
        "runway": "10R",
    }


def test_flights_start_on_the_runway_or_on_final():
    takeoff = flights.start_body("takeoff", "baron")["data"]["runway_start"]
    assert takeoff == {"airport_id": "KPDX", "runway": "10R"}
    final = flights.start_body("landing", "737")["data"]["runway_start"]
    assert final["final_distance_in_nautical_miles"] == flights.FINAL_NM


class _Throttle:
    """A bridge for the throttle hold: an airspeed that the test moves, and
    the throttle it was given."""

    def __init__(self, ias):
        self.ias, self.throttle, self.yoke = ias, [], []

    def read(self, *refs):
        return {flights.IAS: self.ias}

    def write(self, ref, value):
        {flights.THROTTLE: self.throttle, flights.YOKE: self.yoke}[ref].append(value)
        return {"ok": True}


def test_the_throttle_hold_opens_when_slow_and_damps_a_speed_change():
    hold = flights.SpeedHold(throttle_from=0.4)
    hold.target_kt = 100.0
    br = _Throttle(90.0)
    hold.bridge = br
    for t in range(10):  # slow and steady: the throttle opens a little more each tick
        hold.tick(float(t))
    assert br.throttle[0] > 0.4 and br.throttle[-1] > br.throttle[0]
    fast = flights.SpeedHold(throttle_from=0.4)
    fast.target_kt, fast.bridge = 100.0, _Throttle(100.0)
    fast.tick(0.0)
    fast.bridge.ias = 104.0  # gaining 4 kt in a second
    fast.tick(1.0)
    assert fast.bridge.throttle[-1] < 0.4  # damped: back before the error is large
    hold.bridge = _Throttle(None)
    assert hold.tick(11.0) is None  # no airspeed: nothing written


def test_the_controls_are_given_back_at_the_end_of_a_flight():
    hands = flights.new_hands("baron")
    br = _Throttle(100.0)
    hands.speed.target_kt = 100.0
    hands.speed.start(br)
    assert hands.speed.running
    hands.release(br)
    assert not hands.speed.running
    assert br.yoke == [0.0]  # the stick in the middle


def test_a_failed_step_with_an_abort_stops_the_flight_and_turns_back():
    sim = _FlyingSim(["approach"], [1.0])  # the glideslope never captured
    said = []
    sim.say_text = lambda text, ask=False: said.append(text) or {"outcome": "executed"}
    runner = scenario.Runner(sim, voice=None, poll_sec=0.0, log=lambda _line: None)
    gs = "sim/cockpit2/autopilot/glideslope_status"
    results = runner.run(
        [
            scenario.Step(
                "on the glideslope",
                sim=(scenario.equals("captured", gs, 2),),
                abort=flights.TURN_BACK,
            ),
            scenario.Step("gear down", say="gear down"),
            scenario.Step("landing", sim=(scenario.phase_is("landing"),)),
        ]
    )
    assert said == ["heading two eight three", "altitude hold"]  # not "gear down"
    assert [r["step"] for r in results] == [
        "on the glideslope",
        "turn back west",
        "hold the altitude",
        "gear down",
        "landing",
    ]
    assert not results[0]["ok"]
    assert results[3]["skipped"] and results[3]["note"] == "not run: on the glideslope failed"
    assert scenario.line(results[3]) == "· gear down: skipped (not run: on the glideslope failed)"
    assert scenario.summary(results)["failed"] == ["on the glideslope"]


def test_flights_that_end_in_the_air_turn_back_west():
    # Left flying east, the A330 reached the end of the installed scenery.
    for flight in (flights.takeoff, flights.go_around, flights.restart):
        steps = flight("a330", flights.new_hands("a330"))
        assert [s.say for s in steps[-2:]] == ["heading two eight three", "altitude hold"]


# ---- Phase F.4's rules, flown (owner's OK 2026-11-19) ------------------------
def _named(steps, name):
    return next(s for s in steps if s.name == name)


@pytest.mark.parametrize("aircraft", list(flights.PLANES))
def test_the_speed_brake_is_held_on_the_takeoff_roll_then_cancelled(aircraft):
    steps = flights.takeoff(aircraft, flights.new_hands(aircraft))
    names = [s.name for s in steps]
    held = "speed brake on the takeoff roll is held"
    assert (held in names) is flights.PLANES[aircraft].speed_brakes
    if held not in names:
        return
    vr = next(n for n in names if n.startswith("accelerate to Vr"))
    assert names.index("takeoff roll") < names.index(held) < names.index(vr)
    step = _named(steps, held)
    assert step.outcomes == ("safety_hold",)
    assert step.reply == "Confirm speed brake, takeoff roll"
    cancel = steps[names.index(held) + 1]
    assert (cancel.say, cancel.after) == ("cancel", held)
    assert cancel.sim[0].datarefs == (flights.SPEEDBRAKE,)


@pytest.mark.parametrize("flight", [flights.landing, flights.go_around])
@pytest.mark.parametrize("aircraft", list(flights.PLANES))
def test_gear_up_on_the_approach_is_held_then_cancelled(flight, aircraft):
    steps = flight(aircraft, flights.new_hands(aircraft))
    names = [s.name for s in steps]
    held = "gear up on approach is held"
    if not flights.PLANES[aircraft].retractable_gear:
        assert held not in names and "gear down" not in names  # fixed gear
        return
    assert names.index("approach") < names.index(held)  # in the phase first
    assert _named(steps, held).reply == "Confirm gear up, on approach"
    cancel = steps[names.index(held) + 1]
    assert (cancel.say, cancel.after) == ("cancel", held)
    assert cancel.sim[0].datarefs == (flights.GEAR,)


@pytest.mark.parametrize("aircraft", list(flights.PLANES))
def test_flaps_up_when_slow_is_flown_where_the_final_is_slow_enough(aircraft):
    from vimaan_nlu.safety import SafetyContext, evaluate_safety

    p = flights.PLANES[aircraft]
    steps = flights.landing(aircraft, flights.new_hands(aircraft))
    names = [s.name for s in steps]
    held = "flaps up when slow is held"
    # The airliners fly their finals within 10 kt of the clean stall speed in
    # their files (737 145 / 139, A330 150 / 175, MD-82 145 / 150, Citation X
    # 140 / 135); the others' are faster.
    assert (held in names) is (aircraft in ("737", "a330", "md82", "citx"))
    if held not in names:
        return
    step = _named(steps, held)
    # Said only when the rule would hold it, 2 kt to spare: never flaps up
    # on a final that's still slowing.
    assert step.only_if is not None and step.only_if.ok(p.vs_clean_kt + 8)
    assert not step.only_if.ok(p.vs_clean_kt + 8.5)
    # What the plugin's rule says on that final is the reply looked for.
    ctx = SafetyContext(
        agl_ft=1200.0,
        on_ground=False,
        ias_kt=float(p.final_kt),
        vs_fpm=-700.0,
        phase="approach",
        stall_clean_kt=p.vs_clean_kt,
    )
    verdict = evaluate_safety("set_flaps", {"position": "up"}, ctx)
    assert verdict.requires_confirmation and verdict.message == step.reply
    cancel = steps[names.index(held) + 1]
    assert (cancel.say, cancel.after) == ("cancel", held)
    assert cancel.sim[0].datarefs == (flights.FLAPS,)


def test_a_rule_check_not_said_has_no_cancel():
    # The flaps-up step skipped (not slow enough): its "cancel" isn't said.
    class Sim:
        def read(self, refs):
            return dict.fromkeys(refs, 200.0)  # 200 kt: never slow

    hold, cancel = flights._held(
        "flaps up when slow",
        "flaps up",
        "Confirm flaps up, slow",
        scenario.above("flaps still out", flights.FLAPS, 0.3),
        only_if=scenario.below("slow enough", flights.IAS, 147.0),
    )
    runner = scenario.Runner(Sim(), voice=None, poll_sec=0.0, log=lambda _line: None)
    results = runner.run([hold, cancel])
    assert [r.get("skipped") for r in results] == [True, True]


# --- Phase L.1: relative commands --------------------------------------------

from simtest import relative  # noqa: E402
from simtest.tier1 import FLEET  # noqa: E402
from simtest.tier1 import SETUPS as AIR_STARTS  # noqa: E402

# How the model reads each phrase of the checklist (v27's labels).
RELATIVE_NLU = {
    "autopilot on": ("toggle_autopilot", {"state": "on", "instance": "1"}),
    "heading mode": ("set_autopilot_mode", {"mode": "heading"}),
    "altitude hold": ("set_autopilot_mode", {"mode": "altitude hold"}),
    "heading one eight zero": ("set_autopilot_heading", {"degrees": "180"}),
    "turn left twenty degrees": ("turn_by", {"turn": "left", "turn_degrees": "20"}),
    "come right thirty degrees": ("turn_by", {"turn": "right", "turn_degrees": "30"}),
    "turn right one two zero": ("turn_by", {"turn": "right", "turn_degrees": "120"}),
    "turn south": ("set_compass_heading", {"compass": "south"}),
    "climb another thousand": ("change_altitude_by", {"direction": "up", "altitude_change": "1000"}),
    "descend two thousand feet": ("set_autopilot_altitude", {"altitude": "2000"}),
    "down five hundred feet": ("change_altitude_by", {"direction": "down", "altitude_change": "500"}),
    "descend at one thousand five hundred feet per minute": ("set_vertical_speed", {"vs": "1500", "direction": "down"}),
    "altitude six thousand": ("set_autopilot_altitude", {"altitude": "6000"}),
    "speed up ten knots": ("change_speed_by", {"direction": "up", "speed_change": "10"}),
}  # fmt: skip


def _relative_physics(xp):
    """An autopilot that gets there at once: the heading follows the bug, and
    a V/S descent arrives at the selected altitude and holds it. Altitude
    hold caught the aircraft 300 ft above the start altitude, as in the sim
    (2026-11-20): the checklist measures from the altitude held."""
    ap = "sim/cockpit2/autopilot/"
    g = "sim/cockpit2/gauges/indicators/"
    xp.effects["sim/autopilot/vertical_speed"] = lambda x: x.datarefs.update(
        {ap + "vvi_status": 2, ap + "altitude_mode": 4}
    )
    xp.datarefs.update(
        {
            g + "heading_AHARS_deg_mag_pilot": 180.0,
            g + "altitude_ft_pilot": 6300.0,
            g + "airspeed_kts_pilot": 148.0,
            ap + "heading_dial_deg_mag_pilot": 180.0,
        }
    )

    def script(x):
        x.datarefs[g + "heading_AHARS_deg_mag_pilot"] = x.datarefs[
            ap + "heading_dial_deg_mag_pilot"
        ]
        if x.datarefs.get(ap + "vvi_status") == 2:
            x.datarefs[g + "altitude_ft_pilot"] = x.datarefs[ap + "altitude_dial_ft"]
            x.datarefs.update({ap + "vvi_status": 0, ap + "altitude_mode": 6})
            x.datarefs[ap + "altitude_hold_status"] = 2

    return script


def test_the_relative_checklist_passes_on_the_fake_baron(monkeypatch, tmp_path):
    import test_plugin

    test_plugin.NLU.clear()
    for text, (intent, slots) in RELATIVE_NLU.items():
        test_plugin.NLU[text] = (intent, slots, 0.99)
    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    results: list = []
    try:
        monkeypatch.setattr(module, "BRIDGE_POLL_SEC", 0.0)
        monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
        _generic_sim(xp)
        xp.datarefs["sim/flightmodel/position/y_agl"] = 5000 * 0.3048
        script = _relative_physics(xp)
        br = Bridge(tmp_path / "Vimaan_Logs", poll_sec=0.002)
        br.enable()
        plugin._bridge_next = 0.0
        runner = scenario.Runner(br, voice=None, poll_sec=0.005, log=lambda _line: None)
        t = threading.Thread(
            target=lambda: results.extend(runner.run(relative.steps("baron"))), daemon=True
        )
        t.start()
        while t.is_alive():
            script(xp)
            plugin._flight_loop()
            t.join(0.001)
    finally:
        plugin.XPluginStop()
    failed = {r["step"]: r["problems"] for r in results if not r["ok"]}
    assert failed == {}
    assert not any(r.get("skipped") for r in results)


def test_the_relative_checklist_asks_both_questions_on_every_aircraft():
    for aircraft in FLEET:
        steps = relative.steps(aircraft)
        asked = [s for s in steps if s.outcomes == ("ambiguous",)]
        assert [s.say for s in asked] == ["turn right one two zero", "descend two thousand feet"]
        mach = [s for s in steps if s.name == "a change in Mach is refused"]
        assert bool(mach) == AIR_STARTS[aircraft].jet


def test_every_heading_step_lets_the_a330_turn_its_knob():
    # heading_by_knob: the A330 sets a heading over frames ("pending").
    from simtest import flights, tier1

    words = ("heading one", "heading two", "turn left", "turn south", "come right", "one two zero")
    for aircraft in FLEET:
        steps = [*relative.steps(aircraft), *tier1.steps(aircraft), *flights.TURN_BACK]
        for step in steps:
            if step.say and step.say.startswith(words) and "executed" in step.outcomes:
                assert "pending" in step.outcomes, (aircraft, step.name)


def test_relative_needs_a_relative_model():
    assert cli._model_lacks("relative", {"model_intents": ["set_autobrake"]})
    assert cli._model_lacks("relative", {"model_intents": ["turn_by"]}) is None


# --- Your own voice: simtest record, run --recordings ----------------------

import array  # noqa: E402
import math  # noqa: E402
import wave  # noqa: E402

from simtest import record  # noqa: E402


def test_record_lists_each_phrase_of_a_checklist_once():
    said = record.phrases("relative", list(FLEET))
    words = [p for p, _ in said]
    assert len(words) == len(set(words)) == 24
    assert words[:4] == ["autopilot on", "heading mode", "altitude hold", "heading one eight zero"]
    assert {"one two zero", "two thousand", "mach point seven eight"} <= set(words)  # answers, jets
    # The L.2 jets add the SF50's Mach and the Citation X's speed.
    assert {"mach point four five", "speed two zero zero"} <= set(words)
    assert not any(ask for _, ask in said)
    assert any(ask for _, ask in record.phrases("tier4", ["baron"]))  # the question key


class FakeMic:
    """The plugin's recorder, faked: each take is the next of ``takes``."""

    def __init__(self, takes):
        self.takes = list(takes)
        self.error = None

    def start(self):
        pass

    def poll(self):
        return 0

    def stop(self):
        return self.takes.pop(0) if len(self.takes) > 1 else self.takes[0]


def tone(seconds=1.0, amplitude=6000):
    n = int(record.SAMPLE_RATE * seconds)
    s = [int(amplitude * math.sin(2 * math.pi * 220 * i / record.SAMPLE_RATE)) for i in range(n)]
    return array.array("h", s).tobytes()


def scripted(answers):
    """Enter-to-start answers in order; Enter-to-stop is always Enter."""
    answers = list(answers)
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return "" if "recording" in prompt else answers.pop(0)

    return ask, prompts


def test_record_saves_wavs_and_goes_on_where_it_stopped(tmp_path):
    ask, _ = scripted(["", "", "q"])
    lines = []
    missing = record.record("relative", ["baron"], tmp_path, FakeMic([tone()]), ask, lines.append)
    total = len(record.phrases("relative", ["baron"]))
    assert missing == total - 2
    with wave.open(str(tmp_path / "001.wav")) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16_000)
    saved = record.load_recordings(tmp_path)
    assert list(saved) == ["autopilot on", "heading mode"]
    ask, _ = scripted(["q"])
    lines = []
    record.record("relative", ["baron"], tmp_path, FakeMic([tone()]), ask, lines.append)
    assert any(line.startswith(f"[3/{total}]") for line in lines)  # resumed at the third


def test_record_asks_again_for_a_take_with_nothing_in_it(tmp_path):
    ask, _ = scripted(["", "", "q"])
    lines = []
    mic = FakeMic([bytes(32_000), tone()])  # a silent second, then the phrase
    record.record("relative", ["baron"], tmp_path, mic, ask, lines.append)
    assert any("Not kept: too quiet" in line for line in lines)
    assert record.load_manifest(tmp_path)["recordings"]["autopilot on"]["takes"] == 1


def test_record_redoes_the_one_before(tmp_path):
    ask, _ = scripted(["", "r", "", "q"])
    record.record("relative", ["baron"], tmp_path, FakeMic([tone()]), ask, lambda _line: None)
    entry = record.load_manifest(tmp_path)["recordings"]["autopilot on"]
    assert entry["takes"] == 2 and "redo" not in entry


def test_the_runner_plays_the_pilots_recordings(tmp_path):
    class Bridge:
        def __init__(self):
            self.calls = []

        def say_wav(self, path, ask=False):
            self.calls.append(("wav", str(path)))
            return {"ok": True, "heard": "turn south", "outcome": "executed", "replies": []}

        def say_text(self, text, ask=False):
            self.calls.append(("text", text))
            return {"ok": True, "outcome": "executed", "replies": []}

    wav = tmp_path / "001.wav"
    br = Bridge()
    runner = scenario.Runner(
        br, voice=None, recordings={"turn south": wav}, poll_sec=0.0, log=lambda _line: None
    )
    results = runner.run(
        [scenario.Step("recorded", say="turn south"), scenario.Step("not", say="gear down")]
    )
    assert br.calls == [("wav", str(wav)), ("text", "gear down")]
    assert results[0]["heard_ok"] is True and results[1].get("not_recorded") is True
    assert scenario.summary(results)["said_by_voice"] == 1


# ---- X-Plane crashing on the first flight load (2026-11-22) -----------------
def _crashy(monkeypatch, outcomes, alive=False):
    """start_flight raising (or not) per ``outcomes``; what the recovery did."""
    from simtest import xplane

    calls = []
    it = iter(outcomes)

    def start(body):
        calls.append("start")
        exc = next(it)
        if exc:
            raise exc

    monkeypatch.setattr(xplane, "start_flight", start)
    monkeypatch.setattr(xplane, "alive", lambda: alive)
    monkeypatch.setattr(xplane, "close_crash_reporter", lambda: calls.append("close reporter"))
    monkeypatch.setattr(xplane, "launch", lambda: calls.append("launch"))
    monkeypatch.setattr(xplane, "record_crash", lambda *a: calls.append("recorded"))
    monkeypatch.setattr(xplane.time, "sleep", lambda _s: None)
    return xplane, calls, None


def test_a_crash_while_loading_opens_x_plane_again(monkeypatch):
    import http.client

    gone = http.client.RemoteDisconnected("Remote end closed connection")
    xplane, calls, _ = _crashy(monkeypatch, [gone, None])
    said = []
    xplane.start_flight_recovering({}, log=said.append)
    # Only the crash reporter is closed, never X-Plane; then a fresh launch.
    assert calls == ["start", "recorded", "close reporter", "launch", "start"]
    assert "try 2 of 3" in said[0]


def test_an_error_while_x_plane_still_runs_is_not_a_crash(monkeypatch):
    import urllib.error

    xplane, calls, _ = _crashy(monkeypatch, [urllib.error.URLError("timed out")], alive=True)
    monkeypatch.setattr(xplane.time, "monotonic", iter(range(0, 1000, 100)).__next__)
    with pytest.raises(urllib.error.URLError):
        xplane.start_flight_recovering({}, log=lambda _m: None)
    assert calls == ["start"]  # nothing closed, nothing launched


def test_three_crashes_in_a_row_give_up_with_a_clear_message(monkeypatch):
    import http.client

    gone = http.client.RemoteDisconnected("Remote end closed connection")
    xp, calls, _ = _crashy(monkeypatch, [gone, gone, gone])
    with pytest.raises(xp.XPlaneError, match="crashed loading the flight 3 times"):
        xp.start_flight_recovering({}, log=lambda _m: None)
    assert calls.count("launch") == 2 and calls.count("start") == 3


def test_a_crash_is_recorded_with_the_newest_dump(monkeypatch, tmp_path):
    from simtest import xplane

    dumps = tmp_path / "Output" / "crash_reports" / "completed"
    dumps.mkdir(parents=True)
    (dumps / "old.dmp").write_text("x")
    newest = dumps / "new.dmp"
    newest.write_text("x")
    import os

    os.utime(dumps / "old.dmp", (1, 1))
    monkeypatch.setattr(xplane, "FOLDER", tmp_path)
    monkeypatch.setattr(xplane, "CRASH_LOG", tmp_path / "crashes.jsonl")
    body = {"data": {"aircraft": {"path": "Aircraft/x.acf"}}}
    xplane.record_crash(body, 1, ConnectionError("gone"))
    line = json.loads((tmp_path / "crashes.jsonl").read_text())
    assert (line["aircraft"], line["dump"], line["vimaan_installed"]) == (
        "Aircraft/x.acf",
        "new.dmp",
        False,
    )


def test_x_planes_fatal_alert_is_read_from_its_log(tmp_path):
    from simtest import xplane

    (tmp_path / "Log.txt").write_text(
        "0:00:00.000 D/HID: HID Bridge Running\n"
        "0:00:00.000 E/SYS: MACIBM_alert: Fatal Error\n"
        "0:00:00.000 E/SYS: MACIBM_alert: X-Plane must be restarted through Steam.\n"
        "0:00:00.000 E/SYS: MACIBM_alert: Steam must be running in order to run X-Plane.\n"
        "0:00:00.000 E/SYS: MACIBM_alert: /private/var/tmp/builds/hl_init.cpp:1682\n"
    )
    assert xplane.fatal_alert(tmp_path) == (
        "X-Plane must be restarted through Steam. Steam must be running in order to run X-Plane."
    )
    (tmp_path / "Log.txt").write_text("0:00:00.000 I/NET: fine\n")
    assert xplane.fatal_alert(tmp_path) is None


@pytest.mark.parametrize("aircraft", ["c172", "sr22"])
def test_fixed_gear_is_refused_and_never_moved(aircraft):
    steps = flights.takeoff(aircraft, flights.new_hands(aircraft))
    first = _named(steps, "gear up refused (fixed gear)")
    assert (first.say, first.outcomes, first.reply) == ("gear up", ("rejected",), "fixed gear")
    assert "gear up with a positive rate" not in [s.name for s in steps]
    go = flights.go_around(aircraft, flights.new_hands(aircraft))
    assert "gear up" not in [s.name for s in go]


def test_one_engine_restarts_the_engine():
    steps = flights.restart("c172", flights.new_hands("c172"))
    said = [s.say for s in steps if s.say]
    assert said[:2] == ["shut down the engine", "start the engine"]
    assert steps[1].sim[0].datarefs == (f"{flights.RUNNING}[0]",)


def test_the_glideslope_wait_grows_for_slow_approaches():
    assert flights._to_glideslope_sec(flights.PLANES["baron"]) == 150
    assert flights._to_glideslope_sec(flights.PLANES["c172"]) == 203


@pytest.mark.parametrize("aircraft", sorted(flights.PLANES))
def test_each_aircraft_climbs_at_a_rate_it_can_hold(aircraft):
    from vimaan_nlu.normalization import normalize_aviation_input

    p = flights.PLANES[aircraft]
    said = f"climb at {p.climb_words} feet per minute"
    assert normalize_aviation_input(said) == f"climb at {p.climb_fpm} feet per minute"
    steps = flights.takeoff(aircraft, flights.new_hands(aircraft))
    assert _named(steps, "climb on the autopilot").say == said
    go = flights.go_around(aircraft, flights.new_hands(aircraft))
    assert _named(go, "climb away").say == said


def test_the_light_singles_climb_slower_than_1500():
    # At 1,500 fpm the C172 stalled off the autopilot at 650 ft (2026-11-22).
    assert flights.PLANES["c172"].climb_fpm == 500
    assert flights.PLANES["sr22"].climb_fpm <= 700


def test_the_level_off_waits_for_a_slow_climb():
    p = flights.PLANES
    assert flights._climb_sec(p["737"], flights.CRUISE_FT, 300) == 300  # as flown
    assert flights._climb_sec(p["c172"], flights.CRUISE_FT, 300) == 780
    steps = flights.takeoff("c172", flights.new_hands("c172"))
    assert _named(steps, "level at 4000").sim[0].within_sec == 780


# ---- the runner's feet on the runway (2026-11-22) ---------------------------
def _feet(values, origin=(45.588, -122.597)):
    hold = flights.RunwayHold()
    writes = []
    hold.read = lambda ref: values[ref]
    hold.write = lambda ref, v: writes.append((ref, v))
    hold._ids = {
        n: n for n in (flights.PSI, flights.LAT, flights.LON, flights.ON_GROUND, flights.RUDDER)
    }
    hold.heading, hold._origin = flights.RUNWAY_TRUE_DEG, origin
    return hold, writes


def test_the_rudder_holds_right_when_the_nose_swings_left():
    lat0, lon0 = 45.588, -122.597
    values = {flights.PSI: 110.0, flights.LAT: lat0, flights.LON: lon0, flights.ON_GROUND: 1}
    hold, writes = _feet(values)
    assert hold.tick(0.0) > 0  # right rudder for a nose left of the runway
    values[flights.PSI] = 125.0
    assert hold.tick(0.1) < 0
    assert writes[-1][0] == flights.RUDDER
    assert hold.worst_deg == pytest.approx(8.8)


def test_the_distance_off_the_centreline():
    hold, _ = _feet({})
    lat0, lon0 = hold._origin
    h = math.radians(flights.RUNWAY_TRUE_DEG)
    along = 300.0  # metres down the runway
    lat = lat0 + along * math.cos(h) / 111_320.0
    lon = lon0 + along * math.sin(h) / (111_320.0 * math.cos(math.radians(lat0)))
    assert hold.right_of_line_m(lat, lon) == pytest.approx(0.0, abs=0.01)
    # 10 m north of a runway heading east-south-east is left of it.
    assert hold.right_of_line_m(lat0 + 10 / 111_320.0, lon0) < -8


def test_the_takeoff_hold_lets_go_a_second_after_liftoff():
    lat0, lon0 = 45.588, -122.597
    values = {flights.PSI: 118.8, flights.LAT: lat0, flights.LON: lon0, flights.ON_GROUND: 0}
    hold, _ = _feet(values)
    assert hold.tick(10.0) is not None
    assert hold.tick(11.0) is None
    hold._origin = None  # the rollout holds until it is stopped
    assert hold.tick(20.0) is not None and hold.tick(30.0) is not None


def test_the_takeoff_is_steered_and_checked_on_the_runway():
    hands = flights.new_hands("c172")
    steps = flights.takeoff("c172", hands)
    names = [s.name for s in steps]
    thrust = _named(steps, "rudder and takeoff thrust (pilot)")
    assert thrust.write is None and thrust.pilot is not None
    assert names.index("stayed on the runway") == names.index("airborne") + 1
    check = _named(steps, "stayed on the runway").sim[0]
    hands.steer.worst_m = 9.6  # the C172 steered, measured
    assert check.ok(check.value({}))
    hands.steer.worst_m = 42.3  # hands off, measured
    assert not check.ok(check.value({}))
    hands.steer.worst_m = None  # never measured: not a pass
    assert not check.ok(check.value({}))


def test_the_landing_is_steered_on_the_rollout():
    hands = flights.new_hands("c172")
    steps = flights.landing("c172", hands)
    assert steps[-1].name == "rolled out straight"
    check = steps[-1].sim[0]
    hands.steer.worst_deg = 3.0
    assert check.ok(check.value({}))


# The full-flap limits in each aircraft's .acf (simtest.aircraft_file.flap_limits,
# X-Plane 12.4.4, read 2026-11-22).
FULL_FLAPS_KT = {
    "baron": 122,
    "kingair": 150,
    "c172": 85,
    "sr22": 110,
    "citx": 250,
    "md82": 195,
    "sf50": 190,
}


@pytest.mark.parametrize("aircraft", sorted(FULL_FLAPS_KT))
def test_the_approach_is_flown_under_the_full_flap_limit(aircraft):
    # At 90 kt the C172's "flaps full" (85 kt) was rightly held. The singles
    # fly one notch (the SR22's limit 150 kt).
    p = flights.PLANES[aircraft]
    if any("full" in f for f in p.landing_flaps):
        assert p.approach_kt < FULL_FLAPS_KT[aircraft]
    assert p.final_kt < FULL_FLAPS_KT[aircraft]


def test_the_needles_settle_before_the_approach_is_armed():
    for aircraft in flights.PLANES:
        steps = flights.landing(aircraft, flights.new_hands(aircraft))
        names = [s.name for s in steps]
        assert _named(steps, "course 103").settle_sec >= 5
        assert names.index("course 103") < names.index("approach mode")


@pytest.mark.parametrize("aircraft", ["sr22", "sf50"])
def test_the_garmins_hsi_goes_to_nav_1_before_the_approach(aircraft):
    steps = flights.landing(aircraft, flights.new_hands(aircraft))
    names = [s.name for s in steps]
    hsi = _named(steps, "HSI to NAV 1 (pilot)")
    assert hsi.write == (flights.HSI, 0)
    assert (
        names.index("course 103")
        < names.index("HSI to NAV 1 (pilot)")
        < names.index("approach mode")
    )
    assert "HSI to NAV 1 (pilot)" not in [
        s.name for s in flights.landing("c172", flights.new_hands("c172"))
    ]


def test_the_md82_final_settles_before_the_autopilot():
    steps = flights.go_around("md82", flights.new_hands("md82"))
    assert steps[0].name == "the start settles (pilot)" and steps[0].settle_sec == 15
    assert steps[1].name == "autopilot on"
    assert flights.landing("citx", flights.new_hands("citx"))[0].name == "autopilot on"


def test_the_new_aircraft_start_the_approach_in_the_air_on_the_centreline():
    from simtest import xplane

    lat, lon = flights.final_point()
    lat0, lon0 = flights.THRESHOLD_10R
    north = (lat0 - lat) * 111_320.0
    east = (lon0 - lon) * 111_320.0 * math.cos(math.radians(lat0))
    assert math.hypot(north, east) == pytest.approx(8 * 1852, rel=1e-6)
    assert math.degrees(math.atan2(east, north)) % 360 == pytest.approx(flights.RUNWAY_TRUE_DEG)
    body = flights.start_body("landing", "sr22")
    air = body["data"]["lle_air_start"]
    assert air["heading_true"] == flights.RUNWAY_TRUE_DEG
    assert air["speed_in_meters_per_second"] == round(120 * xplane.KNOT, 2)
    assert air["elevation_in_meters"] == round(2531 * xplane.FEET, 1)  # it dives at first
    c172 = flights.start_body("landing", "c172")["data"]["lle_air_start"]
    assert c172["elevation_in_meters"] == round(1531 * xplane.FEET, 1)
    # The first four keep X-Plane's own final start, as flown; takeoffs the runway.
    assert "runway_start" in flights.start_body("landing", "737")["data"]
    assert "runway_start" in flights.start_body("takeoff", "sr22")["data"]


def test_the_approach_waits_for_level_and_aborts_when_it_fails():
    steps = flights.landing("citx", flights.new_hands("citx"))
    names = [s.name for s in steps]
    level = _named(steps, "level")
    assert names.index("altitude hold") < names.index("level") < names.index("the ILS")
    assert level.abort == flights.TURN_BACK
    assert level.sim[0].ok(-250) and not level.sim[0].ok(1500)
    assert _named(steps, "approach").abort == flights.TURN_BACK
    assert _named(steps, "landing").abort == flights.TURN_BACK


@pytest.mark.parametrize("aircraft", ["c172", "sr22"])
def test_the_light_singles_fly_the_coupled_approach_with_one_notch(aircraft):
    assert flights.PLANES[aircraft].landing_flaps == ("flaps down",)


def test_the_jets_that_climb_out_of_the_air_start_start_farther_out():
    lat, lon = flights.final_point(14.0)
    body = flights.start_body("landing", "md82")["data"]["lle_air_start"]
    assert (body["latitude"], body["longitude"]) == (lat, lon)
    assert flights.PLANES["c172"].final_nm == 8.0
    # The glideslope wait covers the longer final: 9.3 nm at 170 kt, plus 45 s.
    assert flights._to_glideslope_sec(flights.PLANES["md82"]) == 242
    assert flights._to_glideslope_sec(flights.PLANES["737"]) == 150  # as flown


def test_the_md82_lands_on_flaps_28():
    assert flights.PLANES["md82"].landing_flaps == ("flaps twenty eight",)


def test_the_md82_is_configured_before_the_glideslope():
    names = [s.name for s in flights.landing("md82", flights.new_hands("md82"))]
    level, ils = names.index("level"), names.index("the ILS")
    for step in ("gear down", "landing flaps (flaps twenty eight)", "speed 145"):
        assert level < names.index(step) < ils
    # The others configure on the glideslope, as flown.
    names = [s.name for s in flights.landing("citx", flights.new_hands("citx"))]
    assert names.index("on the glideslope") < names.index("gear down")
