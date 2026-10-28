"""The real X-Plane plugin, driven against a fake sim (tests/fake_xplane.py).

NLU is stubbed with a lookup table so these run in CI without torch; the
point is the dispatch layer: which datarefs/commands get touched, with what
values, and what the pilot hears.
"""

import importlib.util
import json
import os
import struct
import sys
import threading
import time
import types

import pytest
from fake_xplane import make_sounddevice, make_speech_recognition, make_xp

PLUGIN_PATH = os.path.join(os.path.dirname(__file__), "..", "plugin", "PI_VimaanCoPilot.py")

# utterance -> (intent, slots, confidence)
NLU: dict[str, tuple[str, dict, float]] = {}


def _fake_predict(text, *args, **kwargs):
    intent, slots, conf = NLU.get(text, ("None", {}, 0.99))
    return {
        "intent": intent,
        "slots": dict(slots),
        "confidence": conf,
        "original_text": text,
        "normalized_text": text,
    }


class _FakeLoader:
    def __init__(self, device=None):
        self.device = device
        self.model = self.tokenizer = object()
        self.intent_map_rev: dict = {}
        self.slot_map_rev: dict = {}

    def load_all(self, quantize=False, backend="torch"):
        return {
            "model": {"model_path": "/fake/v11", "device": "cpu"},
            "maps": {"intents": 17, "slots": 13},
            "quantized": quantize,
        }


def _load_plugin(monkeypatch, tmp_path, with_torch=True, with_sounddevice=True):
    monkeypatch.setenv("HOME", str(tmp_path))
    xp = make_xp()
    pkg = types.ModuleType("XPPython3")
    pkg.xp = xp
    monkeypatch.setitem(sys.modules, "XPPython3", pkg)
    monkeypatch.setitem(sys.modules, "XPPython3.xp", xp)
    monkeypatch.setitem(sys.modules, "speech_recognition", make_speech_recognition())
    monkeypatch.setitem(
        sys.modules, "sounddevice", make_sounddevice() if with_sounddevice else None
    )
    if with_torch:
        fake_torch = types.ModuleType("torch")
        fake_torch.device = lambda name: name
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
    else:
        monkeypatch.setitem(sys.modules, "torch", None)  # import raises ImportError
    loader_mod = types.ModuleType("vimaan_nlu.model_loader")
    loader_mod.ModelLoader = _FakeLoader
    infer_mod = types.ModuleType("vimaan_nlu.inference")
    infer_mod.predict = _fake_predict
    monkeypatch.setitem(sys.modules, "vimaan_nlu.model_loader", loader_mod)
    monkeypatch.setitem(sys.modules, "vimaan_nlu.inference", infer_mod)

    spec = importlib.util.spec_from_file_location("PI_VimaanCoPilot_under_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Finish recordings on key-up so tests don't wait out the release tail;
    # test_release_tail_* covers the real behaviour.
    module.RELEASE_TAIL_SEC = 0.0
    plugin = module.PythonInterface()
    plugin.XPluginStart()
    return module, plugin, xp


@pytest.fixture
def sim(monkeypatch, tmp_path):
    NLU.clear()
    module, plugin, xp = _load_plugin(monkeypatch, tmp_path)
    yield module, plugin, xp
    plugin.XPluginStop()


def say(plugin, text, intent=None, slots=None, conf=0.99):
    if intent is not None:
        NLU[text] = (intent, slots or {}, conf)
    xp_spoken_before = len(sys.modules["XPPython3"].xp.spoken)
    now = time.perf_counter()
    plugin._execute_command({"cmd": "c9999", "text": text, "t_press": now, "t_speech_end": now})
    return sys.modules["XPPython3"].xp.spoken[xp_spoken_before:]


def events(plugin, kind=None):
    with open(plugin.tel.events_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    return [r for r in rows if kind is None or r["kind"] == kind]


# ---------------------------------------------------------------------------
# Startup / registration
# ---------------------------------------------------------------------------
def test_startup_registers_and_logs(sim):
    _, plugin, xp = sim
    assert plugin.ready
    assert "vimaan/push_to_talk" in xp.custom_commands
    assert "vimaan/self_test" in xp.custom_commands
    assert len(xp.flight_loops) == 1
    assert xp.spoken[-1] == "Vimaan AI CoPilot ready"
    kinds = [e["kind"] for e in events(plugin)]
    for kind in ("session_start", "microphone_ready", "plugin_start"):
        assert kind in kinds
    mic = events(plugin, "microphone_ready")[0]
    assert mic["using"] == "Fake Mic" and mic["input_devices"] == ["Fake Mic"]
    assert any(e["stage"] == "model_load" for e in events(plugin, "stage"))


def test_missing_torch_keeps_plugin_loaded_and_explains(monkeypatch, tmp_path):
    NLU.clear()
    _, plugin, xp = _load_plugin(monkeypatch, tmp_path, with_torch=False)
    try:
        assert not plugin.ready
        missing = events(plugin, "dependency_missing")
        assert missing and "torch" in missing[0]["missing"]
        assert "could not load its Python packages" in xp.spoken[-1]
        plugin.OnPressCallback(None)
        assert xp.spoken[-1] == "Vimaan is not ready. Check the log."
    finally:
        plugin.XPluginStop()


# ---------------------------------------------------------------------------
# Numeric setters
# ---------------------------------------------------------------------------
def test_heading_writes_cockpit2_dial(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "hdg", "set_autopilot_heading", {"degrees": "270"})
    assert xp.datarefs["sim/cockpit2/autopilot/heading_dial_deg_mag_pilot"] == 270.0
    assert spoken == ["Heading two seven zero"]


def test_altitude_out_of_range_is_not_written(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "alt", "set_autopilot_altitude", {"altitude": "500"})
    assert xp.writes == []
    assert spoken == ["Altitude out of range"]


def test_flight_level_sets_altitude_dial_in_feet(sim):
    _, plugin, xp = sim
    say(plugin, "fl", "set_flight_level", {"flight_level": "350"})
    assert xp.datarefs["sim/cockpit2/autopilot/altitude_dial_ft"] == 35000.0


def test_com_frequency_written_in_khz(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "com", "set_com_frequency", {"com_port": "2", "frequency": "118.75"})
    # 118.750 MHz -> 118750 kHz (the old code wrote 118750000 into a 10 kHz dataref).
    assert xp.datarefs["sim/cockpit2/radios/actuators/com2_frequency_hz_833"] == 118750
    assert spoken[-1].startswith("COM 2 set to")


def test_com_833_spacing_preserved(sim):
    _, plugin, xp = sim
    say(plugin, "com", "set_com_frequency", {"frequency": "118.775"})
    assert xp.datarefs["sim/cockpit2/radios/actuators/com1_frequency_hz_833"] == 118775


def test_com_port_3_rejected_not_silently_com2(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "com", "set_com_frequency", {"com_port": "3", "frequency": "121.5"})
    assert xp.writes == []
    assert "not supported" in spoken[-1]


# ---------------------------------------------------------------------------
# Binary systems
# ---------------------------------------------------------------------------
def test_autopilot_on_uses_dedicated_command(sim):
    _, plugin, xp = sim
    say(plugin, "ap", "toggle_autopilot_1", {"state": "on"})
    assert xp.commands_fired == ["sim/autopilot/servos_on"]


def test_autopilot_off_toggles_only_when_on(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/cockpit2/autopilot/servos_on"] = 1
    say(plugin, "ap", "toggle_autopilot_1", {"state": "off"})
    assert xp.commands_fired == ["sim/autopilot/servos_toggle"]


def test_autopilot_off_when_already_off_does_nothing(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "ap", "toggle_autopilot_1", {"state": "off"})
    assert xp.commands_fired == []
    assert spoken == ["Autopilot 1 already off"]


def test_autopilot_off_with_unreadable_state_never_blind_toggles(sim):
    _, plugin, xp = sim
    del xp.datarefs["sim/cockpit2/autopilot/servos_on"]
    spoken = say(plugin, "ap", "toggle_autopilot_1", {"state": "off"})
    assert xp.commands_fired == []
    assert spoken == ["Cannot read Autopilot 1 state"]


def test_flight_director_written_directly(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/cockpit2/autopilot/flight_director2_mode"] = 2
    say(plugin, "fd", "toggle_flight_director_2", {"state": "off"})
    assert xp.datarefs["sim/cockpit2/autopilot/flight_director2_mode"] == 0


def test_parking_brake_uses_commands(sim):
    _, plugin, xp = sim
    say(plugin, "pb", "toggle_parking_brake", {"state": "on"})
    assert xp.commands_fired == ["sim/flight_controls/park_brake_set"]


def test_flaps_up_retracts_fully(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/cockpit2/controls/flap_handle_request_ratio"] = 0.5
    say(plugin, "flaps", "toggle_flaps", {"state": "up"})
    assert xp.commands_fired == ["sim/flight_controls/flaps_up_full"]


def test_gear_on_fixed_gear_aircraft_rejected(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/aircraft/gear/acf_gear_retract"] = 0
    spoken = say(plugin, "gear", "toggle_landing_gear", {"state": "up"})
    assert xp.commands_fired == []
    assert spoken == ["This aircraft has fixed gear"]


def test_gear_up_airborne_fires_command(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/position/y_agl"] = 1000.0  # ~3300 ft
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})
    assert xp.commands_fired == ["sim/flight_controls/landing_gear_up"]


def test_gear_up_on_ground_needs_confirmation(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "gear", "toggle_landing_gear", {"state": "up"})
    assert xp.commands_fired == []
    assert "Repeat to confirm" in spoken[-1]
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})
    assert xp.commands_fired == ["sim/flight_controls/landing_gear_up"]


# ---------------------------------------------------------------------------
# Engines: starter is held until running
# ---------------------------------------------------------------------------
def test_engine_start_holds_starter_until_running(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "eng", "toggle_engine_1", {"state": "on"})
    assert spoken == ["Starting engine 1"]
    assert "sim/starters/engage_starter_1" in xp.held
    plugin._flight_loop()  # still cranking
    assert "sim/starters/engage_starter_1" in xp.held
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][0] = 1
    plugin._flight_loop()
    assert xp.held == set()
    assert xp.spoken[-1] == "Engine 1 running"
    assert events(plugin, "crank_end")[-1]["result"] == "running"


def test_engine_start_times_out_and_releases_starter(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "STARTER_MAX_CRANK_SEC", 0.0)
    say(plugin, "eng", "toggle_engine_2", {"state": "on"})
    plugin._flight_loop()
    assert xp.held == set()
    assert "did not start" in xp.spoken[-1]


def test_engine_already_running_is_no_op(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][0] = 1
    spoken = say(plugin, "eng", "toggle_engine_1", {"state": "on"})
    assert xp.commands_fired == []
    assert spoken == ["Engine 1 already running"]


def test_plugin_stop_never_leaves_starter_engaged(monkeypatch, tmp_path):
    NLU.clear()
    _, plugin, xp = _load_plugin(monkeypatch, tmp_path)
    say(plugin, "eng", "toggle_engine_1", {"state": "on"})
    plugin.XPluginStop()
    assert xp.held == set()


def test_engine_shutdown_in_flight_needs_confirmation(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][0] = 1
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs["sim/flightmodel/position/y_agl"] = 1500.0
    say(plugin, "eng", "toggle_engine_1", {"state": "off"})
    assert xp.commands_fired == []
    say(plugin, "eng", "toggle_engine_1", {"state": "off"})
    assert xp.commands_fired == ["sim/starters/shut_down_1"]


def test_engine_2_on_single_engine_aircraft_rejected(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/aircraft/engine/acf_num_engines"] = 1
    spoken = say(plugin, "eng", "toggle_engine_2", {"state": "on"})
    assert xp.commands_fired == []
    assert spoken == ["This aircraft has no engine 2"]


# ---------------------------------------------------------------------------
# Decision layer + telemetry
# ---------------------------------------------------------------------------
def test_low_confidence_not_dispatched(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "mumble", "set_autopilot_heading", {"degrees": "270"}, conf=0.3)
    assert xp.writes == []
    assert spoken == ["Please repeat that"]
    assert events(plugin, "command_done")[-1]["outcome"] == "low_confidence"


def test_command_trace_is_logged_end_to_end(sim):
    _, plugin, _ = sim
    say(plugin, "hdg", "set_autopilot_heading", {"degrees": "90"})
    trace = [e for e in events(plugin) if e.get("cmd") == "c9999"]
    kinds = [(e["kind"], e.get("stage")) for e in trace]
    assert ("stage", "nlu") in kinds
    assert ("stage", "dispatch") in kinds
    assert ("sim_write", None) in kinds
    done = trace[-1]
    assert done["kind"] == "command_done" and done["outcome"] == "executed"
    assert "total_ms" in done and "speech_to_action_ms" in done


def test_command_verification_logged_after_delay(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    say(plugin, "ap", "toggle_autopilot_1", {"state": "on"})
    plugin._flight_loop()
    verify = events(plugin, "verify")[-1]
    # The fake sim does not model the servos, so the read-back reports the mismatch.
    assert verify["ok"] is False and verify["expected"] == 1


def test_self_test_checks_intent_and_slots(sim):
    module, plugin, xp = sim
    for text, intent in module.PythonInterface.SELF_TEST_PROBES:
        slots = {s: "1" for s in module.REQUIRED_SLOTS[intent]}
        NLU[text] = (intent, slots, 0.99)
    text, intent = module.PythonInterface.SELF_TEST_PROBES[7]
    NLU[text] = (intent, {}, 0.99)  # e.g. the COM probe losing its frequency
    plugin._run_self_test()
    assert xp.spoken[-1] == "Self test complete. 9 of 10 passed."
    failed = [e for e in events(plugin, "self_test_probe") if not e["ok"]]
    assert failed[0]["missing_slots"] == ["frequency"]
    assert xp.writes == [] and xp.commands_fired == []  # self-test never actuates


def test_push_to_talk_command_binding(sim, monkeypatch):
    module, plugin, xp = sim
    calls = []
    monkeypatch.setattr(plugin, "OnPressCallback", lambda ref, source: calls.append("down"))
    monkeypatch.setattr(plugin, "OnReleaseCallback", lambda ref, source: calls.append("up"))
    handler = xp.custom_commands["vimaan/push_to_talk"]
    assert handler("vimaan/push_to_talk", xp.CommandBegin, None) == 1
    handler("vimaan/push_to_talk", xp.CommandContinue, None)
    handler("vimaan/push_to_talk", xp.CommandEnd, None)
    assert calls == ["down", "up"]


# ---------------------------------------------------------------------------
# Hold-to-talk: press -> record -> release -> STT worker -> flight loop
# ---------------------------------------------------------------------------
LOUD = struct.pack("<2h", 8000, -8000) * 8000  # 1 s of 16 kHz, rms 8000
SILENT = b"\x00\x00" * 16000


def _talk(plugin, audio, transcript="set heading two seven zero"):
    plugin.recognizer.next_result = transcript
    sd = sys.modules["sounddevice"]
    plugin.OnPressCallback(None)
    sd.streams[-1].feed(audio)
    plugin.OnReleaseCallback(None)
    for t in threading.enumerate():
        if t.name.startswith("stt-"):
            t.join(timeout=5)
    plugin._flight_loop()


def test_hold_to_talk_end_to_end(sim):
    _, plugin, xp = sim
    NLU["set heading two seven zero"] = ("set_autopilot_heading", {"degrees": "270"}, 0.99)
    _talk(plugin, LOUD)
    assert xp.datarefs["sim/cockpit2/autopilot/heading_dial_deg_mag_pilot"] == 270.0
    assert sys.modules["sounddevice"].streams[-1].closed  # mic released
    stt = [e for e in events(plugin, "stage") if e["stage"] == "stt"][-1]
    assert stt["text"] == "set heading two seven zero"
    capture = [e for e in events(plugin, "stage") if e["stage"] == "capture"][-1]
    assert capture["audio_sec"] == 1.0
    assert events(plugin, "command_done")[-1]["outcome"] == "executed"


def test_silent_mic_reported(sim):
    _, plugin, xp = sim
    _talk(plugin, SILENT)
    assert xp.spoken[-1] == "Microphone appears silent"
    assert events(plugin, "command_done")[-1]["outcome"] == "silent_mic"


def test_accidental_tap_ignored(sim):
    _, plugin, xp = sim
    _talk(plugin, LOUD[:3200])  # 0.1 s
    assert events(plugin, "command_done")[-1]["outcome"] == "too_short"
    assert xp.writes == []


def test_unrecognised_speech(sim):
    _, plugin, xp = sim
    _talk(plugin, LOUD, transcript=sys.modules["speech_recognition"].UnknownValueError())
    assert xp.spoken[-1] == "I could not understand you"
    assert events(plugin, "stage_ended")[-1]["stage"] == "stt"


def test_key_repeat_does_not_restart_recording(sim):
    _, plugin, _ = sim
    sd = sys.modules["sounddevice"]
    plugin.OnPressCallback(None)
    plugin.OnPressCallback(None)
    assert len(sd.streams) == 1
    plugin.OnReleaseCallback(None)


def test_flight_loop_drains_mic_every_frame_while_recording(sim):
    module, plugin, _ = sim
    sd = sys.modules["sounddevice"]
    assert plugin._flight_loop() == module.QUEUE_POLL_INTERVAL_SEC
    plugin.OnPressCallback(None)
    sd.streams[-1].feed(LOUD)
    assert plugin._flight_loop() == -1.0  # every frame while the key is held
    assert sd.streams[-1].read_available == 0  # drained on the main thread
    plugin.OnReleaseCallback(None)


def test_flight_loop_yields_to_stt_worker(sim, monkeypatch):
    module, plugin, _ = sim
    slept = []
    monkeypatch.setattr(module.time, "sleep", slept.append)
    plugin._transcribing = True
    assert plugin._flight_loop() == -1.0
    assert slept == [module.STT_GIL_YIELD_SEC]
    plugin._transcribing = False


def test_missed_key_up_auto_stops(sim, monkeypatch):
    module, plugin, _ = sim
    monkeypatch.setattr(module, "MAX_RECORD_SEC", -1.0)
    plugin.OnPressCallback(None)
    sys.modules["sounddevice"].streams[-1].feed(LOUD)
    plugin._flight_loop()
    assert plugin._recording is None
    capture = [e for e in events(plugin, "stage") if e["stage"] == "capture"][-1]
    assert capture["released_by"] == "auto_stop"


def test_missing_sounddevice_degrades_to_no_microphone(monkeypatch, tmp_path):
    NLU.clear()
    _, plugin, xp = _load_plugin(monkeypatch, tmp_path, with_sounddevice=False)
    try:
        assert plugin.ready and plugin.recorder is None
        assert events(plugin, "microphone_unavailable")
        plugin.OnPressCallback(None)
        assert xp.spoken[-1] == "Microphone not available"
    finally:
        plugin.XPluginStop()


# ---------------------------------------------------------------------------
# Session report (scripts/vimaan_log_report.py) over a real plugin session
# ---------------------------------------------------------------------------
def _report_module():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "vimaan_log_report.py")
    spec = importlib.util.spec_from_file_location("vimaan_log_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_log_report_summarises_clean_session(monkeypatch, tmp_path):
    NLU.clear()
    _, plugin, _ = _load_plugin(monkeypatch, tmp_path)
    say(plugin, "set heading 270", "set_autopilot_heading", {"degrees": "270"})
    say(plugin, "set altitude 500", "set_autopilot_altitude", {"altitude": "500"})
    plugin.XPluginStop()
    text = _report_module().report(plugin.tel.events_path)
    assert "Ended: clean shutdown" in text
    assert "executed" in text and "rejected" in text
    assert "slot_rejected" in text  # the out-of-range altitude is a listed problem


def test_log_report_flags_abrupt_end(monkeypatch, tmp_path):
    NLU.clear()
    _, plugin, _ = _load_plugin(monkeypatch, tmp_path)
    say(plugin, "gear down", "toggle_landing_gear", {"state": "down"})
    events_path = plugin.tel.events_path
    lines = events_path.read_text().splitlines()
    plugin.XPluginStop()
    # Simulate a crash: drop the shutdown events and tear the last line.
    kept = [ln for ln in lines if '"plugin_stop"' not in ln and '"session_end"' not in ln]
    events_path.write_text("\n".join(kept) + '\n{"kind": "sta')
    text = _report_module().report(events_path)
    assert "ABRUPTLY" in text


XPLANE_ROOT = os.environ.get(
    "VIMAAN_XPLANE_ROOT",
    os.path.expanduser("~/Library/Application Support/Steam/steamapps/common/X-Plane 12"),
)


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(XPLANE_ROOT, "Resources", "plugins", "DataRefs.txt")),
    reason="no local X-Plane install (set VIMAAN_XPLANE_ROOT)",
)
def test_every_sim_name_exists_in_xplane():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "check_xplane_refs.py")
    spec = importlib.util.spec_from_file_location("check_xplane_refs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--xplane", XPLANE_ROOT]) == 0


def test_release_tail_keeps_recording_after_key_up(sim, monkeypatch):
    # PortAudio delivers audio late and pilots release on the last word; the
    # first in-sim run lost "up"/"on"/"off" from every command.
    module, plugin, _ = sim
    monkeypatch.setattr(module, "RELEASE_TAIL_SEC", 0.05)
    sd = sys.modules["sounddevice"]
    plugin.OnPressCallback(None)
    sd.streams[-1].feed(LOUD)
    plugin.OnReleaseCallback(None)
    assert plugin._recording is not None  # still capturing the tail
    sd.streams[-1].feed(LOUD[:3200])  # the late-arriving last word
    plugin.OnPressCallback(None)  # a press during the tail doesn't restart
    assert len(sd.streams) == 1
    time.sleep(0.06)
    plugin._flight_loop()
    assert plugin._recording is None
    capture = [e for e in events(plugin, "stage") if e["stage"] == "capture"][-1]
    assert capture["audio_sec"] == 1.1  # tail included
    for t in threading.enumerate():
        if t.name.startswith("stt-"):
            t.join(timeout=5)


def test_misheard_command_does_not_cancel_pending_confirmation(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][0] = 1
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs["sim/flightmodel/position/y_agl"] = 400.0
    say(plugin, "engine one off", "toggle_engine_1", {"state": "off"})
    assert xp.commands_fired == []
    say(plugin, "engine", "toggle_engine_1", {})  # clipped repeat: rejected
    say(plugin, "engine one off", "toggle_engine_1", {"state": "off"})
    assert xp.commands_fired == ["sim/starters/shut_down_1"]


def test_executed_command_cancels_pending_confirmation(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][0] = 1
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs["sim/flightmodel/position/y_agl"] = 400.0
    say(plugin, "engine one off", "toggle_engine_1", {"state": "off"})
    say(plugin, "hdg", "set_autopilot_heading", {"degrees": "90"})
    say(plugin, "engine one off", "toggle_engine_1", {"state": "off"})
    assert "sim/starters/shut_down_1" not in xp.commands_fired  # asks again
