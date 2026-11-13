"""The real X-Plane plugin, driven against a fake sim (tests/fake_xplane.py).

The voice worker process is replaced by ``FakeWorkerClient``, which runs the
real ``VoiceWorker`` protocol in-process with a fake microphone, a scripted
speech-to-text and an NLU lookup table — so these run in CI without torch,
audio hardware or network. The point is the plugin: which datarefs/commands
get touched, with what values, what the pilot hears, and that nothing heavy
runs on X-Plane's main thread.
"""

import importlib.util
import json
import os
import struct
import sys
import time
import types

import pytest
from fake_xplane import make_sounddevice, make_xp
from vimaan_nlu.audio_capture import PushToTalkRecorder
from vimaan_nlu.voice_worker import VoiceWorker

PLUGIN_PATH = os.path.join(os.path.dirname(__file__), "..", "plugin", "PI_VimaanCoPilot.py")

# utterance -> (intent, slots, confidence)
NLU: dict[str, tuple[str, dict, float]] = {}
# What the fake speech-to-text returns next (a transcript, or an exception to raise).
STT: dict[str, object] = {"next": "set heading two seven zero"}


class UnknownValueError(Exception):
    """Named like speech_recognition's, which the worker classifies by name."""


def _fake_predict(text):
    intent, slots, conf = NLU.get(text, ("None", {}, 0.99))
    return {"intent": intent, "slots": dict(slots), "confidence": conf, "normalized_text": text}


class FakeWorkerClient:
    """In-process stand-in for the worker process (same interface as
    vimaan_nlu.worker_client.WorkerClient)."""

    boot = "ready"  # or "fatal" (model failed to load) / "no_mic"
    sd = None  # the fake sounddevice module

    def __init__(self, argv, log_path, env=None, cwd=None):
        self.argv, self.log_path, self.env, self.cwd = argv, log_path, env, cwd
        self.outbox, self.sent = [], []
        self.alive, self.returncode, self.pid = False, None, 4242

    def start(self):
        self.alive = True
        self.recorder = None if self.boot == "no_mic" else PushToTalkRecorder(sd=self.sd)
        self.worker = VoiceWorker(
            self.outbox.append, self.recorder, self._transcribe, _fake_predict, release_tail_sec=0
        )
        if self.boot == "fatal":
            self.outbox.append({"type": "fatal", "error": "No module named 'torch'"})
        else:
            self.outbox.append(
                {"type": "ready", "load_ms": 12.0, "model_path": "/fake/v11", "mic": {}}
            )

    @staticmethod
    def _transcribe(raw, rate):
        result = STT["next"]
        if isinstance(result, Exception):
            raise result
        return result

    def send(self, msg):
        self.sent.append(msg)
        if not self.alive:
            return False
        self.worker.handle(msg)
        return True

    def poll(self):
        if self.recorder is not None:
            self.recorder.wait_idle()  # the real worker has its own threads
        self.worker.tick()
        out = list(self.outbox)
        self.outbox.clear()  # in place: the worker holds a reference to this list
        return out

    def log_tail(self, lines=20):
        return ["Fatal Python error: Segmentation fault"]

    def stop(self, timeout=2.0):
        self.alive = False


def _load_plugin(monkeypatch, tmp_path, boot="ready"):
    monkeypatch.setenv("HOME", str(tmp_path))
    xp = make_xp()
    pkg = types.ModuleType("XPPython3")
    pkg.xp = xp
    monkeypatch.setitem(sys.modules, "XPPython3", pkg)
    monkeypatch.setitem(sys.modules, "XPPython3.xp", xp)
    # The plugin must never import these into X-Plane: make any import fail.
    for heavy in ("torch", "speech_recognition", "sounddevice", "transformers"):
        monkeypatch.setitem(sys.modules, heavy, None)
    sd = make_sounddevice()

    spec = importlib.util.spec_from_file_location("PI_VimaanCoPilot_under_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = type("Client", (FakeWorkerClient,), {"boot": boot, "sd": sd})
    module.WorkerClient = client
    plugin = module.PythonInterface()
    plugin.XPluginStart()
    plugin.sd = sd
    plugin._flight_loop()  # receive the worker's ready / fatal
    return module, plugin, xp


@pytest.fixture
def sim(monkeypatch, tmp_path):
    NLU.clear()
    STT["next"] = "set heading two seven zero"
    module, plugin, xp = _load_plugin(monkeypatch, tmp_path)
    yield module, plugin, xp
    plugin.XPluginStop()


def say(plugin, text, intent=None, slots=None, conf=0.99):
    """Dispatch a prediction as if the worker had just returned it."""
    if intent is not None:
        NLU[text] = (intent, slots or {}, conf)
    xp_spoken_before = len(sys.modules["XPPython3"].xp.spoken)
    now = time.perf_counter()
    p = _fake_predict(text)
    plugin._execute_command(
        {
            "cmd": "c9999",
            "text": text,
            "prediction": {**p, "nlu_ms": 5.0, "normalized": text},
            "t_press": now,
            "t_release": now,
        }
    )
    return sys.modules["XPPython3"].xp.spoken[xp_spoken_before:]


def events(plugin, kind=None):
    with open(plugin.tel.events_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    return [r for r in rows if kind is None or r["kind"] == kind]


# ---------------------------------------------------------------------------
# Startup / registration / worker lifecycle
# ---------------------------------------------------------------------------
def test_startup_registers_and_logs(sim):
    _, plugin, xp = sim
    assert plugin.ready
    assert "vimaan/push_to_talk" in xp.custom_commands
    assert "vimaan/self_test" in xp.custom_commands
    assert len(xp.flight_loops) == 1
    assert xp.spoken[-1] == "Vimaan AI CoPilot ready"
    kinds = [e["kind"] for e in events(plugin)]
    for kind in ("session_start", "worker_spawn", "plugin_start", "worker_ready"):
        assert kind in kinds
    assert any(e["stage"] == "model_load" for e in events(plugin, "stage"))


def test_plugin_never_imports_heavy_modules(sim):
    # _load_plugin made torch / speech_recognition / sounddevice unimportable;
    # the plugin still loaded and is ready, so none of them run in X-Plane.
    module, plugin, _ = sim
    assert plugin.ready
    for heavy in ("torch", "speech_recognition", "sounddevice", "transformers"):
        assert heavy not in vars(module)


def test_worker_launched_with_xppython3_interpreter(sim):
    _, plugin, xp = sim
    argv = plugin.worker.argv
    assert argv[1:4] == ["-u", "-m", "vimaan_nlu.voice_worker"]
    assert plugin.worker.env["HF_HUB_OFFLINE"] == "1"
    assert plugin.worker.env["PYTHONPATH"].startswith(plugin.worker.cwd)


def test_worker_defaults_to_offline_stt_and_saves_audio(sim):
    _, plugin, _ = sim
    argv = plugin.worker.argv
    assert argv[argv.index("--stt-backend") + 1] == "vosk"
    assert argv[argv.index("--save-audio-dir") + 1] == str(plugin.tel.log_dir / "audio")
    assert argv[argv.index("--audio-prefix") + 1] == f"{plugin.tel.session_id}_"


def test_audio_saving_can_be_turned_off(sim, monkeypatch):
    _, plugin, _ = sim
    monkeypatch.setattr(plugin, "SAVE_AUDIO", False)
    assert "--save-audio-dir" not in plugin._worker_argv()


def test_stt_fallback_logged(sim):
    _, plugin, _ = sim
    plugin._on_worker_message(
        {
            "type": "ready",
            "stt_backend": "google",
            "stt": {
                "fallback_from": "vosk",
                "error": "ModuleNotFoundError: No module named 'vosk'",
            },
        }
    )
    assert plugin.STT_BACKEND == "google"
    (ev,) = events(plugin, "stt_fallback")
    assert ev["requested"] == "vosk" and ev["using"] == "google" and "vosk" in ev["error"]


def test_overlay_shows_the_command(sim):
    _, plugin, xp = sim
    assert events(plugin, "overlay_start")[-1]["backend"] == "native"
    STT["next"] = "set heading two seven zero"
    NLU["set heading two seven zero"] = ("set_autopilot_heading", {"degrees": "270"}, 0.99)
    plugin.OnPressCallback(None)
    plugin._flight_loop()
    plugin._on_worker_message({"type": "partial", "cmd": plugin._active["cmd"], "text": "set head"})
    plugin._flight_loop()
    assert xp.draw_windows() == ["REC", "set head"]
    plugin.sd.streams[-1].feed(b"\x40\x1f\xc0\xe0" * 8000)
    plugin.OnReleaseCallback(None)
    plugin._flight_loop()
    screen = xp.draw_windows()
    assert screen[:2] == ["HEARD", '"set heading two seven zero"']
    assert "MEANT" in screen and "Heading   hdg 270" in screen
    assert "DONE" in screen and "Heading two seven zero" in screen


def test_overlay_toggle_cycles_modes(sim):
    _, plugin, xp = sim
    handler = xp.custom_commands["vimaan/toggle_overlay"]
    handler(None, xp.CommandBegin, None)
    plugin._flight_loop()
    assert plugin.overlay_state.mode == "pinned" and xp.draw_windows() == [
        "VIMAAN",
        "Overlay pinned",
    ]
    handler(None, xp.CommandBegin, None)
    plugin._flight_loop()
    assert plugin.overlay_state.mode == "off" and xp.draw_windows() == []
    assert [e["mode"] for e in events(plugin, "overlay_mode")] == ["pinned", "off"]


def test_overlay_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("VIMAAN_OVERLAY", "off")
    module, plugin, xp = _load_plugin(monkeypatch, tmp_path)
    try:
        assert plugin.overlay is None and xp.windows == {}
    finally:
        plugin.XPluginStop()


def test_worker_uses_xp_python_executable_when_available(sim, monkeypatch):
    _, plugin, xp = sim
    monkeypatch.setattr(xp, "pythonExecutable", sys.executable, raising=False)
    assert plugin._python_executable() == sys.executable


def test_model_load_failure_keeps_plugin_loaded_and_explains(monkeypatch, tmp_path):
    NLU.clear()
    _, plugin, xp = _load_plugin(monkeypatch, tmp_path, boot="fatal")
    try:
        assert not plugin.ready
        assert events(plugin, "worker_fatal")
        assert "could not load its model" in xp.spoken[-1]
        plugin.OnPressCallback(None)
        assert xp.spoken[-1] == "Vimaan is not ready yet"
    finally:
        plugin.XPluginStop()


def test_worker_crash_is_logged_and_restarted(sim):
    _, plugin, xp = sim
    plugin.worker.alive, plugin.worker.returncode = False, -11
    plugin._flight_loop()
    crash = events(plugin, "worker_exit")[-1]
    assert crash["returncode"] == -11 and "Segmentation fault" in crash["log_tail"][-1]
    assert "Voice engine restarting" in xp.spoken
    plugin._flight_loop()  # the new worker reports ready
    assert plugin.ready and len(events(plugin, "worker_spawn")) == 2


def test_worker_restarts_are_bounded(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "WORKER_MAX_RESTARTS", 0)
    plugin.worker.alive = False
    plugin._flight_loop()
    assert xp.spoken[-1] == "Voice engine failed. Check the log."
    assert plugin.worker is None and not plugin.ready


def test_plugin_stop_stops_worker(monkeypatch, tmp_path):
    _, plugin, _ = _load_plugin(monkeypatch, tmp_path)
    worker = plugin.worker
    plugin.XPluginStop()
    assert not worker.alive


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
def test_autopilot_on_presses_the_checked_toggle(sim):
    _, plugin, xp = sim  # state read as off, so the toggle turns it on
    say(plugin, "ap", "toggle_autopilot", {"state": "on"})
    assert xp.commands_fired == ["sim/autopilot/servos_toggle"]


def test_autopilot_on_with_unreadable_state_uses_the_explicit_command(sim):
    _, plugin, xp = sim  # a blind toggle could turn it off
    del xp.datarefs["sim/cockpit2/autopilot/servos_on"]
    say(plugin, "ap", "toggle_autopilot", {"state": "on"})
    assert xp.commands_fired == ["sim/autopilot/servos_on"]


@pytest.mark.parametrize(("instance", "suffix"), [("1", ""), ("2", "2")])
def test_autopilot_engages_on_an_aircraft_that_ignores_servos_on(
    sim, monkeypatch, instance, suffix
):
    # In flight the A330 ignored servos_on (and servos2_on) but honours the
    # panel toggle. Simulate exactly that and check the read-back passes.
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    dataref = f"sim/cockpit2/autopilot/servos{suffix}_on"
    xp.effects[f"sim/autopilot/servos{suffix}_toggle"] = lambda x: x.datarefs.__setitem__(
        dataref, 1 - x.datarefs[dataref]
    )
    assert say(plugin, "ap", "toggle_autopilot", {"state": "on", "instance": instance}) == [
        f"Autopilot {instance} on"
    ]
    plugin._flight_loop()
    assert xp.datarefs[dataref] == 1
    assert events(plugin, "verify")[-1]["ok"] is True


def test_autopilot_off_toggles_only_when_on(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/cockpit2/autopilot/servos_on"] = 1
    say(plugin, "ap", "toggle_autopilot", {"state": "off"})
    assert xp.commands_fired == ["sim/autopilot/servos_toggle"]


def test_autopilot_off_when_already_off_does_nothing(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "ap", "toggle_autopilot", {"state": "off"})
    assert xp.commands_fired == []
    assert spoken == ["Autopilot 1 already off"]


def test_autopilot_off_with_unreadable_state_never_blind_toggles(sim):
    _, plugin, xp = sim
    del xp.datarefs["sim/cockpit2/autopilot/servos_on"]
    spoken = say(plugin, "ap", "toggle_autopilot", {"state": "off"})
    assert xp.commands_fired == []
    assert spoken == ["Cannot read Autopilot 1 state"]


def test_flight_director_written_directly(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/cockpit2/autopilot/flight_director2_mode"] = 2
    say(plugin, "fd", "toggle_flight_director", {"state": "off", "instance": "2"})
    assert xp.datarefs["sim/cockpit2/autopilot/flight_director2_mode"] == 0


def test_parking_brake_uses_commands(sim):
    _, plugin, xp = sim
    say(plugin, "parking brake", "toggle_parking_brake", {"state": "on"})
    assert xp.commands_fired == ["sim/flight_controls/park_brake_set"]


def test_flaps_up_retracts_fully(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/cockpit2/controls/flap_handle_request_ratio"] = 0.5
    say(plugin, "flaps", "set_flaps", {"position": "up"})
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
    assert spoken == ["Confirm gear up, low altitude. Say override, or repeat to confirm."]
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})
    assert xp.commands_fired == ["sim/flight_controls/landing_gear_up"]


# ---------------------------------------------------------------------------
# Override and cancel
# ---------------------------------------------------------------------------
GEAR_UP = "sim/flight_controls/landing_gear_up"


@pytest.mark.parametrize("word", ["override", "confirm", "affirmative"])
def test_override_executes_the_held_command(sim, word):
    _, plugin, xp = sim
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})  # held: on the ground
    assert say(plugin, word) == ["Override", "Gear up"]
    assert xp.commands_fired == [GEAR_UP]
    override = events(plugin, "safety_override")[-1]
    assert (
        override["rule"] == "gear_up_low_altitude" and override["intent"] == "toggle_landing_gear"
    )
    assert events(plugin, "command_done")[-1]["outcome"] == "executed"


def test_override_with_nothing_held(sim):
    _, plugin, xp = sim
    assert say(plugin, "override") == ["Nothing to override"]
    assert xp.commands_fired == []


def test_override_after_the_window_does_nothing(sim, monkeypatch):
    _, plugin, xp = sim
    monkeypatch.setattr(plugin._confirm_gate, "_window", -1.0)
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})
    assert say(plugin, "override") == ["Nothing to override"]
    assert xp.commands_fired == []


def test_override_skips_only_the_rule_it_overrides(sim, monkeypatch):
    module, plugin, xp = sim
    # Another rule that also applies to gear up: overriding the first must
    # not wave this one through.
    real = module.evaluate_safety

    def two_rules(intent, slots, ctx, skip=()):
        verdict = real(intent, slots, ctx, skip=skip)
        if verdict.requires_confirmation or "second_rule" in skip:
            return verdict
        from vimaan_nlu.safety import SafetyVerdict

        return SafetyVerdict(True, "Confirm second rule", "second_rule")

    monkeypatch.setattr(module, "evaluate_safety", two_rules)
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})
    assert say(plugin, "override") == [
        "Override",
        "Confirm second rule. Say override, or repeat to confirm.",
    ]
    assert xp.commands_fired == []
    say(plugin, "override")
    assert xp.commands_fired == [GEAR_UP]


def test_cancel_drops_the_held_command(sim):
    _, plugin, xp = sim
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})
    assert say(plugin, "cancel") == ["Cancelled"]
    assert events(plugin, "command_done")[-1]["outcome"] == "cancelled"
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})  # held again, not confirmed
    assert xp.commands_fired == []
    assert say(plugin, "override") == ["Override", "Gear up"]


def test_cancel_closes_the_to_or_two_question(sim):
    _, plugin, xp = sim
    say(
        plugin,
        "set altitude two one thousand feet",
        "set_autopilot_altitude",
        {"altitude": "21000"},
    )
    assert say(plugin, "negative") == ["Cancelled"]
    assert say(plugin, "one thousand") == ["Say the full command"]
    assert xp.writes == []


def test_cancel_with_nothing_pending(sim):
    _, plugin, _ = sim
    assert say(plugin, "cancel") == ["Nothing to cancel"]


def test_validator_refusals_cannot_be_overridden(sim):
    _, plugin, xp = sim
    say(plugin, "heading", "set_autopilot_heading", {"degrees": "450"})
    assert say(plugin, "override") == ["Nothing to override"]
    assert xp.writes == []


# ---------------------------------------------------------------------------
# Engines: starter is held until running
# ---------------------------------------------------------------------------
def test_engine_start_holds_starter_until_running(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "engine", "toggle_engine", {"state": "on"})
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
    say(plugin, "engine", "toggle_engine", {"state": "on", "instance": "2"})
    plugin._flight_loop()
    assert xp.held == set()
    assert "did not start" in xp.spoken[-1]


def test_engine_already_running_is_no_op(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][0] = 1
    spoken = say(plugin, "engine", "toggle_engine", {"state": "on"})
    assert xp.commands_fired == []
    assert spoken == ["Engine 1 already running"]


def test_plugin_stop_never_leaves_starter_engaged(monkeypatch, tmp_path):
    NLU.clear()
    _, plugin, xp = _load_plugin(monkeypatch, tmp_path)
    say(plugin, "engine", "toggle_engine", {"state": "on"})
    plugin.XPluginStop()
    assert xp.held == set()


def test_engine_shutdown_in_flight_needs_confirmation(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][0] = 1
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs["sim/flightmodel/position/y_agl"] = 1500.0
    say(plugin, "engine", "toggle_engine", {"state": "off"})
    assert xp.commands_fired == []
    say(plugin, "engine", "toggle_engine", {"state": "off"})
    assert xp.commands_fired == ["sim/starters/shut_down_1"]


def test_engine_2_on_single_engine_aircraft_rejected(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/aircraft/engine/acf_num_engines"] = 1
    spoken = say(plugin, "engine", "toggle_engine", {"state": "on", "instance": "2"})
    assert xp.commands_fired == []
    assert spoken == ["This aircraft has no engine 2"]


def test_both_engines_start_together(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "both engines", "toggle_engine", {"state": "on", "instance": "both"})
    assert spoken == ["Starting engine 1", "Starting engine 2"]
    assert xp.held == {"sim/starters/engage_starter_1", "sim/starters/engage_starter_2"}
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][:2] = [1, 1]
    plugin._flight_loop()
    assert xp.held == set()


def test_both_engines_on_single_engine_aircraft(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/aircraft/engine/acf_num_engines"] = 1
    spoken = say(plugin, "both engines", "toggle_engine", {"state": "on", "instance": "both"})
    assert spoken == ["This aircraft has one engine"] and xp.held == set()


# ---------------------------------------------------------------------------
# Engine start by engine type (from X-Plane's acf_en_type, never the name)
# ---------------------------------------------------------------------------
STARTER_1 = "sim/starters/engage_starter_1"
FUEL_LEVER = "sim/cockpit2/engine/actuators/mixture_ratio"
N1 = "sim/cockpit2/engine/indicators/N1_percent"
N2 = "sim/cockpit2/engine/indicators/N2_percent"
RUNNING = "sim/flightmodel/engine/ENGN_running"


def load_aircraft(plugin, xp, engine_type, commands=(), live=None):
    """Switch the fake sim to another aircraft and let the plugin notice.
    Its scripts' datarefs (Override.live) come with its commands unless
    ``live`` says otherwise."""
    from vimaan_nlu.aircraft import OVERRIDES

    xp.datarefs["sim/aircraft/prop/acf_en_type"] = [engine_type] * 16
    for name in commands:
        xp.custom_commands[name] = None
    if live is None:
        live = {d for o in OVERRIDES if set(o.requires) <= set(commands) for d in o.live}
    for name in live:
        xp.datarefs[name] = 0
    xp.aircraft_model = ("other.acf", f"/Aircraft/Other {engine_type}/other.acf")
    plugin._next_state_watch = 0.0
    plugin._flight_loop()
    return plugin.profile


def test_jet_start_opens_fuel_once_the_core_turns(sim):
    _, plugin, xp = sim
    assert load_aircraft(plugin, xp, 7).engine_start == "jet"
    xp.datarefs[FUEL_LEVER][0] = 0.0  # fuel cutoff, as the A330 was in flight
    assert say(plugin, "engine", "toggle_engine", {"state": "on"}) == ["Starting engine 1"]
    plugin._flight_loop()
    assert xp.datarefs[FUEL_LEVER][0] == 0.0  # N2 still 0: no fuel yet
    xp.datarefs[N2][0] = 22.0
    plugin._flight_loop()
    assert xp.datarefs[FUEL_LEVER][0] == 1.0
    assert events(plugin, "fuel_on")[-1]["method"] == "write"
    xp.datarefs[RUNNING][0] = 1
    xp.datarefs[N2][0] = 35.0  # lit, but below the starter cutout: keep cranking
    plugin._flight_loop()
    assert STARTER_1 in xp.held
    xp.datarefs[N2][0] = 58.0
    plugin._flight_loop()
    assert xp.held == set() and xp.spoken[-1] == "Engine 1 running"
    end = events(plugin, "crank_end")[-1]
    assert end["start_type"] == "jet" and end["fuel_on"] is True and end["peak_core_pct"] == 58.0


def test_jet_start_uses_the_aircraft_fuel_lever_when_it_has_one(sim):
    _, plugin, xp = sim
    levers = (
        "laminar/B738/engines/condition_lever_1_idle",
        "laminar/B738/engines/condition_lever_2_idle",
    )
    profile = load_aircraft(plugin, xp, 7, commands=levers)
    assert "b738_condition_levers" in profile.overrides
    xp.datarefs[FUEL_LEVER][0] = 0.0
    say(plugin, "engine", "toggle_engine", {"state": "on"})
    xp.datarefs[N2][0] = 25.0
    plugin._flight_loop()
    assert levers[0] in xp.commands_fired
    assert events(plugin, "fuel_on")[-1]["method"] == levers[0]


def test_jet_that_never_spools_says_what_to_check(sim, monkeypatch):
    module, plugin, xp = sim
    load_aircraft(plugin, xp, 7)
    monkeypatch.setattr(module, "TURBINE_MAX_CRANK_SEC", 0.0)
    say(plugin, "engine", "toggle_engine", {"state": "on"})
    plugin._flight_loop()
    assert xp.held == set()
    assert xp.spoken[-1] == (
        "Engine 1 did not spool up. N2 reached 0 percent. Check bleed air and the engine start switch."
    )


def test_jet_that_spools_but_never_lights(sim):
    _, plugin, xp = sim
    load_aircraft(plugin, xp, 7)
    say(plugin, "engine", "toggle_engine", {"state": "on"})
    xp.datarefs[N2][0] = 24.0
    plugin._flight_loop()
    plugin._cranks["Engine 1"]["max_sec"] = 0.0  # this start's limit runs out
    plugin._flight_loop()
    assert xp.spoken[-1] == (
        "Engine 1 did not light. N2 reached 24 percent. Check fuel and ignition."
    )


def test_turboprop_start_watches_the_gas_generator(sim):
    _, plugin, xp = sim
    assert load_aircraft(plugin, xp, 9).engine_start == "turboprop"  # King Air C90B
    xp.datarefs[FUEL_LEVER][0] = 0.0
    say(plugin, "engine", "toggle_engine", {"state": "on"})
    xp.datarefs[N1][0] = 14.0  # past the 13% at which a PT6 gets fuel
    plugin._flight_loop()
    assert xp.datarefs[FUEL_LEVER][0] == 1.0


def test_electric_engine_has_no_starter(sim):
    _, plugin, xp = sim
    load_aircraft(plugin, xp, 3)
    assert say(plugin, "engine", "toggle_engine", {"state": "on"}) == [
        "Engine 1 is electric and has no starter"
    ]
    assert xp.held == set()


def test_aircraft_profile_logged_with_what_xplane_reports(sim):
    _, plugin, _ = sim
    (event,) = events(plugin, "aircraft")
    assert event["icao"] == "BE58" and event["key"] == "Beechcraft_Baron_58__Baron_58"
    assert event["engine_kind"] == "piston" and event["engines"] == 2
    assert event["flap_notch_degrees"] == [0.0, 13.0, 30.0]
    assert event["v_speeds"]["vfe"] == 152.0 and event["overrides"] == []


def test_switching_aircraft_rebuilds_the_profile(sim):
    _, plugin, xp = sim
    load_aircraft(plugin, xp, 7)
    kinds = [e["engine_kind"] for e in events(plugin, "aircraft")]
    assert kinds == ["piston", "jet"]


def test_shutting_down_engine_two_does_not_confirm_engine_one(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][:2] = [1, 1]
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs["sim/flightmodel/position/y_agl"] = 1500.0
    say(plugin, "engine one", "toggle_engine", {"state": "off", "instance": "1"})
    say(plugin, "engine two", "toggle_engine", {"state": "off", "instance": "2"})
    assert xp.commands_fired == []  # each still waiting for its own repeat


def test_autopilot_two_uses_its_own_binding(sim):
    _, plugin, xp = sim
    say(plugin, "autopilot", "toggle_autopilot", {"state": "on", "instance": "2"})
    assert xp.commands_fired == ["sim/autopilot/servos2_toggle"]


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
    say(plugin, "ap", "toggle_autopilot", {"state": "on"})
    plugin._flight_loop()
    verify = events(plugin, "verify")[-1]
    # The fake sim does not model the servos, so the read-back reports the mismatch.
    assert verify["ok"] is False and verify["expected"] == 1
    # ...and the pilot hears it (in-sim the A330 ignored "autopilot on").
    assert xp.spoken[-1] == "Autopilot 1 did not respond"


@pytest.mark.parametrize(
    ("text", "intent", "slots"),
    [
        ("70", "set_flight_level", {"flight_level": "70"}),  # in-sim: "seventy"
        ("city 8 on", "toggle_flight_director", {"state": "on"}),  # in-sim
        ("on", "toggle_parking_brake", {"state": "on"}),
    ],
)
def test_fragment_without_command_word_not_executed(sim, text, intent, slots):
    _, plugin, xp = sim
    spoken = say(plugin, text, intent, slots)
    assert spoken == ["Please repeat that"]
    assert xp.writes == [] and xp.commands_fired == []
    assert events(plugin, "implausible")[-1]["reason"] == "no_command_word"


@pytest.mark.parametrize("value", ["361", "450", "2040"])
def test_out_of_range_heading_never_reaches_the_sim(sim, value):
    _, plugin, xp = sim
    spoken = say(plugin, "heading", "set_autopilot_heading", {"degrees": value})
    assert spoken == ["Invalid heading value"] and xp.writes == []


@pytest.mark.parametrize(
    ("value", "written"), [("000", 0.0), ("080", 80.0), ("80", 80.0), ("360", 360.0)]
)
def test_valid_heading_boundaries_written(sim, value, written):
    _, plugin, xp = sim
    say(plugin, "heading", "set_autopilot_heading", {"degrees": value})
    assert xp.writes[-1] == ("sim/cockpit2/autopilot/heading_dial_deg_mag_pilot", written)


def test_com_with_no_port_spoken_tunes_com_1(sim):
    _, plugin, xp = sim  # the behaviour the gate scores: documented default
    say(plugin, "tune", "set_com_frequency", {"frequency": "121.5"})
    assert xp.writes[-1] == ("sim/cockpit2/radios/actuators/com1_frequency_hz_833", 121500)


# ---------------------------------------------------------------------------
# "to" vs "two": the five wrong commands from the v17 test flight
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("heard", "intent", "slots", "question"),
    [
        (
            "set altitude two one thousand feet",
            "set_autopilot_altitude",
            {"altitude": "21000"},
            "Altitude 1000 or 21000 feet? Say it again without the word to.",
        ),
        (
            "set altitude two five thousand feet",
            "set_autopilot_altitude",
            {"altitude": "25000"},
            "Altitude 5000 or 25000 feet? Say it again without the word to.",
        ),
        (
            "set heading two twenty degrees",
            "set_autopilot_heading",
            {"degrees": "220"},
            "Heading zero two zero or two two zero? Say all three digits.",
        ),
        (
            "set heading two ten degrees",
            "set_autopilot_heading",
            {"degrees": "210"},
            "Heading zero one zero or two one zero? Say all three digits.",
        ),
        (
            "flight level two forty",
            "set_flight_level",
            {"flight_level": "240"},
            "Flight level zero four zero or two four zero? Say all three digits.",
        ),
    ],
)
def test_ambiguous_target_asks_and_changes_nothing(sim, heard, intent, slots, question):
    _, plugin, xp = sim
    assert say(plugin, heard, intent, slots) == [question]
    assert xp.writes == [] and xp.commands_fired == []
    assert events(plugin, "command_done")[-1]["outcome"] == "ambiguous"
    assert events(plugin, "ambiguous")[-1]["intent"] == intent


@pytest.mark.parametrize(
    ("heard", "intent", "slots", "dataref", "value"),
    [
        (
            "heading two seven zero",
            "set_autopilot_heading",
            {"degrees": "270"},
            "sim/cockpit2/autopilot/heading_dial_deg_mag_pilot",
            270.0,
        ),
        (
            "set altitude one thousand feet",
            "set_autopilot_altitude",
            {"altitude": "1000"},
            "sim/cockpit2/autopilot/altitude_dial_ft",
            1000.0,
        ),
        (
            "set altitude twenty one thousand feet",
            "set_autopilot_altitude",
            {"altitude": "21000"},
            "sim/cockpit2/autopilot/altitude_dial_ft",
            21000.0,
        ),
    ],
)
def test_clear_answers_set_the_target(sim, heard, intent, slots, dataref, value):
    _, plugin, xp = sim
    say(plugin, heard, intent, slots)
    assert xp.writes[-1] == (dataref, value)


# ---------------------------------------------------------------------------
# Bare-number replies to "to or two?" (2nd v17 flight: "one thousand" and
# "zero" were ignored without a word)
# ---------------------------------------------------------------------------
ALT_DIAL = "sim/cockpit2/autopilot/altitude_dial_ft"
HDG_DIAL = "sim/cockpit2/autopilot/heading_dial_deg_mag_pilot"
ASK_ALT = ("set altitude two one thousand feet", "set_autopilot_altitude", {"altitude": "21000"})
ASK_HDG = ("set heading two twenty degrees", "set_autopilot_heading", {"degrees": "220"})


@pytest.mark.parametrize(
    ("question", "reply", "dataref", "value"),
    [
        (ASK_ALT, "one thousand", ALT_DIAL, 1000.0),  # c0004
        (ASK_ALT, "twenty one thousand feet", ALT_DIAL, 21000.0),
        (ASK_HDG, "zero two zero", HDG_DIAL, 20.0),
        (ASK_HDG, "two two zero", HDG_DIAL, 220.0),
    ],
)
def test_reply_with_an_offered_value_sets_it(sim, question, reply, dataref, value):
    _, plugin, xp = sim
    say(plugin, *question)
    say(plugin, reply)
    assert xp.writes == [(dataref, value)]
    assert events(plugin, "command_done")[-1]["outcome"] == "executed"
    answered = events(plugin, "answered")[-1]
    assert answered["intent"] == question[1] and answered["reply_to"] == "c9999"


def test_reply_naming_neither_value_changes_nothing(sim):
    _, plugin, xp = sim
    say(plugin, *ASK_HDG)
    assert say(plugin, "zero") == ["Say it with the word heading"]  # c0010
    assert xp.writes == []
    assert events(plugin, "implausible")[-1]["reason"] == "not_offered"
    # The question stays open: a clear reply still works.
    say(plugin, "zero two zero")
    assert xp.writes == [(HDG_DIAL, 20.0)]


def test_reply_with_the_same_doubt_asks_again(sim):
    _, plugin, xp = sim
    question = say(plugin, *ASK_HDG)
    assert say(plugin, "two twenty") == question  # 220, or "to twenty"
    assert xp.writes == []


def test_bare_number_with_no_question_is_answered(sim):
    _, plugin, xp = sim
    assert say(plugin, "one thousand", "set_autopilot_altitude", {"altitude": "1000"}) == [
        "Say the full command"
    ]
    assert xp.writes == []
    assert events(plugin, "implausible")[-1]["reason"] == "bare_number"


def test_reply_after_the_window_is_not_taken(sim, monkeypatch):
    _, plugin, xp = sim
    monkeypatch.setattr(plugin, "QUESTION_WINDOW_SEC", 0.0)
    say(plugin, *ASK_ALT)
    assert say(plugin, "one thousand") == ["Say the full command"]
    assert xp.writes == []


def test_executed_command_closes_the_question(sim):
    _, plugin, xp = sim
    say(plugin, *ASK_ALT)
    say(plugin, "altitude one thousand feet", "set_autopilot_altitude", {"altitude": "1000"})
    assert say(plugin, "twenty one thousand") == ["Say the full command"]
    assert xp.writes == [(ALT_DIAL, 1000.0)]


def test_altitude_to_below_the_minimum_asks(sim):
    _, plugin, xp = sim
    # 2nd v17 flight c0013: said "altitude to five hundred", 2,500 was set.
    assert say(
        plugin, "altitude two five hundred", "set_autopilot_altitude", {"altitude": "2500"}
    ) == ["Altitude 500 or 2500 feet? Say it again without the word to."]
    assert xp.writes == []
    assert say(plugin, "five hundred") == ["Altitude out of range"]
    assert xp.writes == []
    say(plugin, "two thousand five hundred")
    assert xp.writes == [(ALT_DIAL, 2500.0)]


def test_flight_level_read_back_keeps_the_leading_zero(sim):
    _, plugin, _ = sim
    # 2nd v17 flight: "flight level zero four zero" was read back "four zero".
    assert say(
        plugin, "flight level zero four zero", "set_flight_level", {"flight_level": "040"}
    ) == ["Flight level zero four zero"]


# ---------------------------------------------------------------------------
# Delayed read-back, and target vs what the aircraft did
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("port", "dataref", "other"),
    [
        (
            "1",
            "sim/cockpit2/radios/actuators/com1_frequency_hz_833",
            "sim/cockpit2/radios/actuators/com2_frequency_hz_833",
        ),
        (
            "2",
            "sim/cockpit2/radios/actuators/com2_frequency_hz_833",
            "sim/cockpit2/radios/actuators/com1_frequency_hz_833",
        ),
    ],
)
def test_com_value_is_checked_again_later(sim, monkeypatch, port, dataref, other):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    before = xp.datarefs[other]
    say(plugin, "com", "set_com_frequency", {"frequency": "118.75", "com_port": port})
    assert xp.datarefs[dataref] == 118750 and xp.datarefs[other] == before
    plugin._flight_loop()
    verify = events(plugin, "verify")[-1]
    assert verify["label"] == f"COM {port}" and verify["ok"] is True and verify["actual"] == 118750


def test_radio_that_does_not_keep_the_value_is_reported(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    say(plugin, "com", "set_com_frequency", {"frequency": "118.75", "com_port": "1"})
    xp.datarefs["sim/cockpit2/radios/actuators/com1_frequency_hz_833"] = 122800
    plugin._flight_loop()
    assert events(plugin, "verify")[-1]["ok"] is False
    assert xp.spoken[-1] == "COM 1 did not respond"


def test_heading_360_held_as_0_is_not_a_failure(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    say(plugin, "heading", "set_autopilot_heading", {"degrees": "360"})
    xp.datarefs["sim/cockpit2/autopilot/heading_dial_deg_mag_pilot"] = 0.0
    plugin._flight_loop()
    assert events(plugin, "verify")[-1]["ok"] is True


def test_aircraft_recorded_after_a_target_changes(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "FOLLOW_UP_SEC", (0.0, 0.0))
    xp.datarefs["sim/cockpit2/gauges/indicators/altitude_ft_pilot"] = 4200.0
    xp.datarefs["sim/cockpit2/autopilot/altitude_hold_status"] = 2
    say(plugin, "altitude", "set_autopilot_altitude", {"altitude": "5000"})
    first = events(plugin, "aircraft_state")[0]
    assert first["after"] == "Altitude" and first["altitude_target"] == 5000.0
    assert first["altitude_actual"] == 4200.0 and first["altitude_hold_status"] == 2


def test_autopilot_change_vimaan_did_not_make_is_logged(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "STATE_WATCH_SEC", 0.0)
    plugin._next_state_watch = 0.0  # the fixture's first loop already armed it
    plugin._flight_loop()
    xp.datarefs["sim/cockpit2/autopilot/servos_on"] = 1
    xp.datarefs["sim/cockpit2/autopilot/heading_mode"] = 1
    plugin._flight_loop()
    change = events(plugin, "autopilot_state_change")[-1]
    assert change["changed"] == {"servos_on": [0, 1], "lateral_mode": [0, 1]}
    assert change["last_vimaan_cmd"] is None


def test_autopilot_change_names_the_command_that_ran_last(sim, monkeypatch):
    # 2nd v17 flight: only target writes counted, so "autopilot on" 0.6 s
    # before a change was logged as "c0019, 357 s earlier".
    module, plugin, xp = sim
    monkeypatch.setattr(module, "STATE_WATCH_SEC", 0.0)
    plugin._next_state_watch = 0.0
    say(plugin, "altitude", "set_autopilot_altitude", {"altitude": "5000"})
    plugin._flight_loop()
    say(plugin, "ap", "toggle_autopilot", {"state": "on"})
    xp.datarefs["sim/cockpit2/autopilot/servos_on"] = 1
    plugin._flight_loop()
    change = events(plugin, "autopilot_state_change")[-1]
    assert change["last_vimaan_intent"] == "toggle_autopilot"
    assert change["sec_since_vimaan_action"] < 1.0


def test_aircraft_type_logged_once(sim, monkeypatch):
    module, plugin, _ = sim
    monkeypatch.setattr(module, "STATE_WATCH_SEC", 0.0)
    plugin._next_state_watch = 0.0
    plugin._flight_loop()
    plugin._flight_loop()
    assert [e["icao"] for e in events(plugin, "aircraft")] == ["BE58"]


# ---------------------------------------------------------------------------
# Autopilot-off diagnostics (vimaan_nlu.disconnect)
# ---------------------------------------------------------------------------
SERVOS = "sim/cockpit2/autopilot/servos_on"
AP_ON = "sim/cockpit2/autopilot/autopilot_on"


@pytest.fixture
def flying(sim, monkeypatch):
    """Autopilot on, state watch every loop, every stick reading kept."""
    module, plugin, xp = sim
    monkeypatch.setattr(module, "STATE_WATCH_SEC", 0.0)
    plugin._stick = module.StickTrace(every_sec=0.0)
    xp.datarefs[SERVOS] = xp.datarefs[AP_ON] = 1
    plugin._next_state_watch = 0.0
    plugin._flight_loop()  # the watch sees the autopilot on
    return module, plugin, xp


def drop(xp):
    xp.datarefs[SERVOS] = xp.datarefs[AP_ON] = 0


def test_a_stick_pull_that_drops_the_autopilot_is_called_out(flying):
    # Flight 203741: the Baron went from level to +1,250 ft/min in ALT HOLD,
    # then the autopilot dropped, with nothing in the log to say why.
    _, plugin, xp = flying
    xp.datarefs["sim/joystick/yoke_pitch_ratio"] = 0.62
    plugin._flight_loop()
    xp.datarefs["sim/joystick/yoke_pitch_ratio"] = 0.05
    drop(xp)
    before = len(xp.spoken)
    plugin._flight_loop()
    assert xp.spoken[before:] == ["Autopilot disconnected. Stick moved."]
    (e,) = events(plugin, "autopilot_disconnect")
    assert e["cause"] == "stick" and e["stick_peak"]["pitch"] == 0.62
    assert e["detail"] == "stick pitch +0.62" and e["nullzone"] == 0.0
    assert e["changed"]["servos_on"] == [1, 0]
    assert events(plugin, "speak")[-1]["cmd"] is None  # not a reply to a command


def test_a_disconnect_button_is_named(flying):
    _, plugin, xp = flying
    xp.effects["sim/autopilot/servos_fdir_off"] = drop
    xp.commandOnce("sim/autopilot/servos_fdir_off")  # a yoke button, not Vimaan
    before = len(xp.spoken)
    plugin._flight_loop()
    assert xp.spoken[before:] == ["Autopilot disconnected by a switch or button."]
    (e,) = events(plugin, "autopilot_disconnect")
    assert e["cause"] == "command"
    assert [(c["command"], c["by_vimaan"]) for c in e["commands"]] == [
        ("sim/autopilot/servos_fdir_off", False)
    ]


def test_autopilot_off_from_vimaan_is_not_called_out_again(flying):
    _, plugin, xp = flying
    xp.effects["sim/autopilot/servos_toggle"] = drop
    spoken = say(plugin, "autopilot off", "toggle_autopilot", {"state": "off"})
    assert spoken == ["Autopilot 1 off"]
    before = len(xp.spoken)
    plugin._flight_loop()
    assert xp.spoken[before:] == []
    (e,) = events(plugin, "autopilot_disconnect")
    assert e["cause"] == "vimaan" and e["commands"][0]["by_vimaan"] is True


def test_no_power_and_unknown_causes(flying):
    _, plugin, xp = flying
    xp.datarefs["sim/cockpit2/autopilot/autopilot_has_power"] = 0
    drop(xp)
    before = len(xp.spoken)
    plugin._flight_loop()
    assert xp.spoken[before:] == ["Autopilot disconnected. The autopilot has no power."]
    xp.datarefs["sim/cockpit2/autopilot/autopilot_has_power"] = 1
    xp.datarefs[SERVOS] = xp.datarefs[AP_ON] = 1
    plugin._flight_loop()
    drop(xp)
    before = len(xp.spoken)
    plugin._flight_loop()
    assert xp.spoken[before:] == ["Autopilot disconnected."]


def test_a_new_flight_is_not_a_disconnect(flying):
    _, plugin, xp = flying
    plugin.XPluginReceiveMessage(0, xp.MSG_PLANE_LOADED, 0)
    drop(xp)
    plugin._flight_loop()  # the reload reset the baseline: no change to report
    xp.datarefs[SERVOS] = xp.datarefs[AP_ON] = 1
    plugin._flight_loop()
    drop(xp)
    before = len(xp.spoken)
    plugin._flight_loop()  # within the quiet window after the reload
    assert xp.spoken[before:] == []
    assert [e["cause"] for e in events(plugin, "autopilot_disconnect")] == ["reload"]


def test_the_joystick_setup_is_logged_per_aircraft(sim):
    _, plugin, _ = sim
    (e,) = events(plugin, "joystick")
    assert e["has_joystick"] == 1 and e["nullzone"] == 0.0
    assert e["stick"] == {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}


def test_watch_handlers_never_swallow_the_command(sim):
    _, plugin, xp = sim
    assert xp.hooks["sim/autopilot/servos_off_any"]
    assert plugin._on_off_command("sim/autopilot/servos_off_any", xp.CommandBegin, "x") == 1
    plugin.XPluginStop()
    assert not any(xp.hooks.values())
    plugin.XPluginStop = lambda: None  # the fixture stops it again


def test_one_digit_heading_asks_again(sim):
    _, plugin, xp = sim
    # in-sim: "heading one eighty" with the last word lost -> heading 001
    spoken = say(plugin, "heading 1", "set_autopilot_heading", {"degrees": "1"})
    assert spoken == ["Say the heading as three digits"]
    assert xp.writes == []
    say(plugin, "heading 090", "set_autopilot_heading", {"degrees": "090"})
    assert xp.writes[-1] == ("sim/cockpit2/autopilot/heading_dial_deg_mag_pilot", 90.0)


def test_self_test_checks_intent_and_slots(sim):
    module, plugin, xp = sim
    for text, intent in module.PythonInterface.SELF_TEST_PROBES:
        slots = {s: "1" for s in module.REQUIRED_SLOTS[intent]}
        NLU[text] = (intent, slots, 0.99)
    text, intent = next(
        p for p in module.PythonInterface.SELF_TEST_PROBES if p[1] == "set_com_frequency"
    )
    NLU[text] = (intent, {}, 0.99)  # e.g. the COM probe losing its frequency
    plugin._run_self_test()
    plugin._flight_loop()  # worker reply
    total = len(module.PythonInterface.SELF_TEST_PROBES)
    assert xp.spoken[-1] == f"Self test complete. {total - 1} of {total} passed."
    failed = [e for e in events(plugin, "self_test_probe") if not e["ok"]]
    assert failed[0]["missing_slots"] == ["frequency"]
    assert xp.writes == [] and xp.commands_fired == []  # self-test never actuates
    (stage,) = [e for e in events(plugin, "stage") if e["stage"] == "self_test"]
    assert 0 <= stage["ms"] < 5000  # was measured from 0 (in-sim: 376928216 ms)


def test_self_test_skips_commands_the_model_was_not_trained_on(sim):
    # v19 predates Tier 2: its self-test covers the commands it knows.
    module, plugin, xp = sim
    probes = module.PythonInterface.SELF_TEST_PROBES
    known = {intent for _, intent in probes if not intent.startswith(("swap_", "set_nav", "set_t"))}
    known -= {"set_com_standby", "set_course"}
    plugin._on_worker_message({"type": "ready", "intents": sorted(known | {"None"})})
    for text, intent in probes:
        NLU[text] = (intent, {s: "1" for s in module.REQUIRED_SLOTS[intent]}, 0.99)
    plugin._run_self_test()
    plugin._flight_loop()
    sent = plugin.worker.sent[-1]["probes"]
    assert {intent for _, intent in sent} == known and len(sent) < len(probes)
    assert xp.spoken[-1] == f"Self test complete. {len(sent)} of {len(sent)} passed."
    (stage,) = [e for e in events(plugin, "stage") if e["stage"] == "self_test"]
    assert stage["not_in_model"] == len(probes) - len(sent) == 8


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
# Hold-to-talk through the worker protocol
# ---------------------------------------------------------------------------
LOUD = struct.pack("<2h", 8000, -8000) * 8000  # 1 s of 16 kHz, rms 8000
SILENT = b"\x00\x00" * 16000


def _talk(plugin, audio, transcript="set heading two seven zero"):
    STT["next"] = transcript
    plugin.OnPressCallback(None)
    plugin._flight_loop()  # mic open in the worker
    plugin.sd.streams[-1].feed(audio)
    plugin.OnReleaseCallback(None)
    plugin._flight_loop()  # reply -> dispatch


def test_hold_to_talk_end_to_end(sim):
    _, plugin, xp = sim
    NLU["set heading two seven zero"] = ("set_autopilot_heading", {"degrees": "270"}, 0.99)
    _talk(plugin, LOUD)
    assert xp.datarefs["sim/cockpit2/autopilot/heading_dial_deg_mag_pilot"] == 270.0
    assert [m["op"] for m in plugin.worker.sent if m["op"] in ("start", "stop")] == [
        "start",
        "stop",
    ]
    stt = [e for e in events(plugin, "stage") if e["stage"] == "stt"][-1]
    assert stt["text"] == "set heading two seven zero"
    capture = [e for e in events(plugin, "stage") if e["stage"] == "capture"][-1]
    assert capture["audio_sec"] == 1.0
    done = events(plugin, "command_done")[-1]
    assert done["outcome"] == "executed" and "speech_to_action_ms" in done
    plugin.worker.recorder.wait_idle()
    assert plugin.sd.streams[-1].closed  # mic released


def test_press_and_release_handlers_only_send_messages(sim):
    _, plugin, _ = sim
    plugin.OnPressCallback(None)
    plugin.OnReleaseCallback(None)
    press = events(plugin, "ptt_press")[-1]
    release = events(plugin, "ptt_release")[-1]
    assert press["handler_ms"] < 5 and release["handler_ms"] < 5
    plugin._flight_loop()


def test_silent_mic_reported(sim):
    _, plugin, xp = sim
    _talk(plugin, SILENT)
    assert xp.spoken[-1] == "Microphone appears silent"
    assert events(plugin, "command_done")[-1]["outcome"] == "silent_mic"


def test_accidental_tap_ignored_quietly(sim):
    _, plugin, xp = sim
    spoken_before = list(xp.spoken)
    _talk(plugin, LOUD[:3200])  # 0.1 s
    assert events(plugin, "command_done")[-1]["outcome"] == "too_short"
    assert xp.writes == [] and xp.spoken == spoken_before


def test_unrecognised_speech(sim):
    _, plugin, xp = sim
    _talk(plugin, LOUD, transcript=UnknownValueError())
    assert xp.spoken[-1] == "I could not understand you"
    assert events(plugin, "stage_ended")[-1]["stage"] == "stt"


def test_key_repeat_sends_one_start(sim):
    _, plugin, _ = sim
    plugin.OnPressCallback(None)
    plugin.OnPressCallback(None)
    assert [m["op"] for m in plugin.worker.sent].count("start") == 1
    plugin.OnReleaseCallback(None)
    plugin._flight_loop()


def test_press_while_previous_command_processing_is_ignored(sim):
    _, plugin, _ = sim
    plugin.OnPressCallback(None)
    plugin.OnReleaseCallback(None)
    plugin.OnPressCallback(None)  # reply not read yet
    assert events(plugin, "ptt_ignored")[-1]["reason"] == "still_processing"
    plugin._flight_loop()


def test_no_reply_times_out(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "WORKER_REPLY_TIMEOUT_SEC", -1.0)
    plugin.OnPressCallback(None)
    monkeypatch.setattr(plugin.worker, "send", lambda msg: True)  # stop is lost
    plugin.OnReleaseCallback(None)
    plugin._flight_loop()
    assert xp.spoken[-1] == "Voice engine timed out"
    assert plugin._active is None


def test_flight_loop_runs_every_frame_only_while_a_reply_is_due(sim):
    module, plugin, _ = sim
    assert plugin._flight_loop() == module.QUEUE_POLL_INTERVAL_SEC
    plugin.OnPressCallback(None)
    assert plugin._flight_loop() == -1.0
    plugin.OnReleaseCallback(None)
    plugin._flight_loop()
    assert plugin._flight_loop() == module.QUEUE_POLL_INTERVAL_SEC


def test_no_microphone_in_worker(monkeypatch, tmp_path):
    NLU.clear()
    _, plugin, xp = _load_plugin(monkeypatch, tmp_path, boot="no_mic")
    try:
        assert plugin.ready
        plugin.OnPressCallback(None)
        plugin._flight_loop()
        assert xp.spoken[-1] == "Microphone error"
        assert events(plugin, "command_done")[-1]["outcome"] == "mic_error"
    finally:
        plugin.XPluginStop()


def test_reply_for_an_abandoned_command_is_ignored(sim):
    _, plugin, xp = sim
    plugin._on_worker_message(
        {
            "type": "result",
            "cmd": "c0999",
            "text": "gear up",
            "intent": "toggle_landing_gear",
            "confidence": 1.0,
            "slots": {"state": "up"},
        }
    )
    assert xp.commands_fired == []


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


def test_misheard_command_does_not_cancel_pending_confirmation(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][0] = 1
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs["sim/flightmodel/position/y_agl"] = 400.0
    say(plugin, "engine one off", "toggle_engine", {"state": "off"})
    assert xp.commands_fired == []
    say(plugin, "engine", "toggle_engine", {})  # clipped repeat: rejected
    say(plugin, "engine one off", "toggle_engine", {"state": "off"})
    assert xp.commands_fired == ["sim/starters/shut_down_1"]


def test_executed_command_cancels_pending_confirmation(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][0] = 1
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs["sim/flightmodel/position/y_agl"] = 400.0
    say(plugin, "engine one off", "toggle_engine", {"state": "off"})
    say(plugin, "hdg", "set_autopilot_heading", {"degrees": "90"})
    say(plugin, "engine one off", "toggle_engine", {"state": "off"})
    assert "sim/starters/shut_down_1" not in xp.commands_fired  # asks again


# ---------------------------------------------------------------------------
# Aircraft probe (pilot-paced, one step per key press)
# ---------------------------------------------------------------------------
AP = "sim/cockpit2/autopilot/"
BARO = "sim/cockpit2/gauges/actuators/barometer_setting_in_hg_pilot"


def press_probe(plugin, xp):
    before = len(xp.spoken)
    plugin._on_probe_cmd(None, xp.CommandBegin, None)
    return xp.spoken[before:]


def settle(plugin, xp, loops=3):
    before = len(xp.spoken)
    for _ in range(loops):
        plugin._flight_loop()
    return xp.spoken[before:]


@pytest.fixture
def probing(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module.probe, "SETTLE_SEC", 0.0)
    return module, plugin, xp


def airborne(xp, autopilot=True):
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs["sim/flightmodel/position/y_agl"] = 1000.0  # ~3,300 ft
    xp.datarefs["sim/flightmodel/position/groundspeed"] = 70.0
    xp.datarefs[AP + "servos_on"] = int(autopilot)
    xp.datarefs[AP + "heading_mode"], xp.datarefs[AP + "heading_status"] = 1, 2
    xp.datarefs[AP + "altitude_mode"], xp.datarefs[AP + "altitude_hold_status"] = 6, 2


def toggle_heading(x):
    on = x.datarefs[AP + "heading_mode"] == 1
    x.datarefs[AP + "heading_mode"], x.datarefs[AP + "heading_status"] = (0, 0) if on else (1, 2)


def go_to(plugin, role):
    plugin._probe["index"] = [s.role for s in plugin._probe["steps"]].index(role)


def test_probe_needs_a_stable_aircraft(probing):
    _, plugin, xp = probing
    xp.datarefs["sim/flightmodel/position/groundspeed"] = 8.0  # taxiing
    assert press_probe(plugin, xp) == [
        "The probe needs the aircraft stopped on the ground, or airborne above fifteen hundred feet"
    ]
    assert plugin._probe is None


def test_ground_probe_measures_and_saves(probing):
    _, plugin, xp = probing
    xp.datarefs["sim/flightmodel/controls/parkbrake"] = 1.0
    xp.effects["sim/instruments/barometer_std"] = lambda x: x.datarefs.__setitem__(BARO, 29.92)
    assert press_probe(plugin, xp) == [
        "Probe ready. 3 ground steps. Press again for altimeter standard."
    ]
    press_probe(plugin, xp)
    assert settle(plugin, xp) == ["Altimeter standard: works. Next, altimeter setting."]
    assert xp.datarefs[BARO] == pytest.approx(30.12)  # put back
    press_probe(plugin, xp)
    assert settle(plugin, xp) == ["Altimeter setting: works. Next, flap handle."]
    press_probe(plugin, xp)
    assert settle(plugin, xp) == ["Flap handle: works.", "Probe complete. Results saved."]
    assert xp.datarefs["sim/cockpit2/controls/flap_handle_request_ratio"] == 0.0
    saved = json.loads(
        (plugin.tel.log_dir / "aircraft" / "Beechcraft_Baron_58__Baron_58.json").read_text()
    )
    assert saved["steps"] == {"baro_std": "works", "dial_baro": "works", "flap_handle": "works"}
    assert saved["aircraft"]["icao"] == "BE58" and saved["end"] == "complete"
    assert plugin.profile.supports("flap_handle") is True


def test_ground_steps_need_the_parking_brake(probing):
    _, plugin, xp = probing
    press_probe(plugin, xp)
    assert press_probe(plugin, xp) == ["Set the parking brake first"]
    assert xp.commands_fired == []


def test_air_probe_presses_a_mode_and_puts_it_back(probing):
    _, plugin, xp = probing
    airborne(xp)
    xp.effects["sim/autopilot/heading"] = toggle_heading
    assert press_probe(plugin, xp)[0].startswith("Probe ready. 15 air steps.")
    press_probe(plugin, xp)
    assert settle(plugin, xp) == ["Heading mode: works. Next, nav mode."]
    assert xp.commands_fired == ["sim/autopilot/heading", "sim/autopilot/heading"]
    assert xp.datarefs[AP + "heading_mode"] == 1


def test_restoring_presses_the_previous_mode_when_toggling_back_isnt_enough(probing):
    # V/S pressed again drops to pitch mode, not back to ALT hold: the probe
    # then presses altitude hold.
    _, plugin, xp = probing
    airborne(xp)

    def vertical_speed(x):
        x.datarefs[AP + "altitude_mode"] = 3 if x.datarefs[AP + "altitude_mode"] == 4 else 4

    xp.effects["sim/autopilot/vertical_speed"] = vertical_speed
    xp.effects["sim/autopilot/altitude_hold"] = lambda x: x.datarefs.__setitem__(
        AP + "altitude_mode", 6
    )
    press_probe(plugin, xp)
    go_to(plugin, "mode_vertical_speed")
    press_probe(plugin, xp)
    assert settle(plugin, xp, loops=4) == ["Vertical speed mode: works. Next, level change."]
    assert xp.commands_fired == [
        "sim/autopilot/vertical_speed",
        "sim/autopilot/vertical_speed",
        "sim/autopilot/altitude_hold",
    ]
    assert xp.datarefs[AP + "altitude_mode"] == 6


def test_no_change_and_missing_controls(probing):
    _, plugin, xp = probing
    airborne(xp)
    xp.missing_commands.add("sim/autopilot/vnav")
    press_probe(plugin, xp)
    go_to(plugin, "mode_nav")
    press_probe(plugin, xp)
    assert settle(plugin, xp) == ["Nav mode: no change. Next, approach mode."]
    go_to(plugin, "mode_vnav")
    assert press_probe(plugin, xp) == ["V NAV: not on this aircraft. Next, autothrottle."]
    assert "sim/autopilot/vnav" not in xp.commands_fired


def test_probe_stops_when_it_cannot_restore(probing):
    _, plugin, xp = probing
    airborne(xp)

    def heading_off_for_good(x):
        x.datarefs[AP + "heading_mode"], x.datarefs[AP + "heading_status"] = 0, 0

    xp.effects["sim/autopilot/heading"] = heading_off_for_good
    press_probe(plugin, xp)
    press_probe(plugin, xp)
    assert settle(plugin, xp, loops=4) == [
        "Probe stopped. After heading mode, the lateral mode did not go back. "
        "Check the autopilot. The probe key continues with nav mode."
    ]
    assert plugin._probe is None
    step = [e for e in events(plugin, "probe_step") if e["role"] == "mode_heading"][-1]
    assert step["restored"] is False and step["left_changed"]["lateral_mode"] == [1, 0]


def test_air_steps_need_the_autopilot_on(probing):
    _, plugin, xp = probing
    airborne(xp, autopilot=False)
    press_probe(plugin, xp)
    assert press_probe(plugin, xp) == ["Turn the autopilot on first"]
    assert xp.commands_fired == []


def test_probe_stop_saves_what_was_measured(probing):
    _, plugin, xp = probing
    airborne(xp)
    press_probe(plugin, xp)
    go_to(plugin, "mode_nav")
    press_probe(plugin, xp)
    settle(plugin, xp)
    plugin._on_probe_stop_cmd(None, xp.CommandBegin, None)
    assert (
        xp.spoken[-1] == "Probe stopped. Results saved. The probe key continues with approach mode."
    )
    assert plugin.profile.supports("mode_nav") is False


def test_a_saved_probe_is_used_the_next_time_the_aircraft_loads(probing):
    _, plugin, xp = probing
    folder = plugin.tel.log_dir / "aircraft"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Other_7__other.json").write_text(
        json.dumps({"steps": {"mode_vertical_speed": "no_change", "mode_heading": "works"}})
    )
    profile = load_aircraft(plugin, xp, 7)
    assert profile.supports("mode_vertical_speed") is False
    assert profile.supports("mode_heading") is True


def test_probe_error_stops_it_cleanly(probing, monkeypatch):
    _, plugin, xp = probing
    airborne(xp)
    press_probe(plugin, xp)
    press_probe(plugin, xp)
    monkeypatch.setattr(plugin, "_probe_state", lambda: 1 / 0)
    assert settle(plugin, xp, loops=1) == ["Probe stopped by an error. Check the autopilot."]
    assert plugin._probe is None


def test_std_toggle_is_pressed_again_when_restoring(probing):
    # 737-style STD: a toggle that remembers the setting.
    _, plugin, xp = probing
    xp.datarefs["sim/flightmodel/controls/parkbrake"] = 1.0
    remembered = {}

    def std_toggle(x):
        if "qnh" in remembered:
            x.datarefs[BARO] = remembered.pop("qnh")
        else:
            remembered["qnh"] = x.datarefs[BARO]
            x.datarefs[BARO] = 29.92

    xp.effects["sim/instruments/barometer_std"] = std_toggle
    press_probe(plugin, xp)
    press_probe(plugin, xp)
    assert settle(plugin, xp)[0] == "Altimeter standard: works. Next, altimeter setting."
    assert xp.commands_fired.count("sim/instruments/barometer_std") == 2
    assert remembered == {} and xp.datarefs[BARO] == pytest.approx(30.12)


# ---------------------------------------------------------------------------
# Tier 1 commands: speed, vertical speed, modes, autothrottle, altimeter, flaps
# ---------------------------------------------------------------------------
SPEED_DIAL = AP + "airspeed_dial_kts_mach"
VVI_DIAL = AP + "vvi_dial_fpm"
FLAP_HANDLE = "sim/cockpit2/controls/flap_handle_request_ratio"
A333 = ("laminar/A333/autopilot/speed_knob_pull", "laminar/A333/autopilot/a_thr_toggle")
B738 = (
    "laminar/B738/engines/condition_lever_1_idle",
    "laminar/B738/engines/condition_lever_2_idle",
    "laminar/B738/EFIS_control/capt/push_button/std_press",
)


def aircraft(plugin, xp, engine_type=7, commands=(), **datarefs):
    for name, value in datarefs.items():
        xp.datarefs[name] = value
    return load_aircraft(plugin, xp, engine_type, commands=commands)


def test_speed_in_knots(sim):
    _, plugin, xp = sim
    assert say(plugin, "speed", "set_autopilot_speed", {"speed": "160"}) == ["Speed one six zero"]
    assert xp.datarefs[SPEED_DIAL] == 160.0 and xp.commands_fired == []


def test_mach_switches_the_dial_to_mach_first(sim):
    _, plugin, xp = sim

    def to_mach(x):
        x.datarefs[AP + "airspeed_is_mach"] = 1 - x.datarefs[AP + "airspeed_is_mach"]

    xp.effects["sim/autopilot/knots_mach_toggle"] = to_mach
    assert say(plugin, "mach", "set_autopilot_speed", {"mach": "0.78"}) == [
        "Mach point seven eight"
    ]
    assert xp.commands_fired == ["sim/autopilot/knots_mach_toggle"]
    assert xp.datarefs[SPEED_DIAL] == pytest.approx(0.78)


@pytest.mark.parametrize(
    ("value", "spoken"),
    [
        ("70", "Speed 70 is below this aircraft's stall speed"),  # Baron Vso 74
        ("240", "Speed 240 is above this aircraft's never exceed speed"),  # Vne 223
    ],
)
def test_speed_outside_the_aircraft_envelope_is_refused(sim, value, spoken):
    _, plugin, xp = sim
    assert say(plugin, "speed", "set_autopilot_speed", {"speed": value}) == [spoken]
    assert xp.writes == []


def test_airbus_pulls_the_speed_knob_for_a_selected_speed(sim):
    _, plugin, xp = sim
    aircraft(plugin, xp, commands=A333)
    say(plugin, "speed", "set_autopilot_speed", {"speed": "160"})
    assert xp.commands_fired == ["laminar/A333/autopilot/speed_knob_pull"]


def test_speed_two_eighty_on_a_jet_is_not_asked(sim):
    _, plugin, xp = sim  # 80 kt is below a 737's stall speed: only 280 makes sense
    aircraft(plugin, xp, **{"sim/aircraft/view/acf_Vso": 110.0, "sim/aircraft/view/acf_Vne": 340.0})
    NLU["speed two eighty"] = ("set_autopilot_speed", {"speed": "280"}, 0.99)
    assert say(plugin, "speed two eighty") == ["Speed two eight zero"]


def test_speed_two_eighty_on_the_baron_asks(sim):
    _, plugin, xp = sim  # Vso 74: "to eighty" is flyable, so both readings stand
    NLU["speed two eighty"] = ("set_autopilot_speed", {"speed": "280"}, 0.99)
    assert say(plugin, "speed two eighty") == ["Speed 80 or 280? Say it again without the word to."]


def test_vertical_speed_engages_the_mode_then_sets_the_rate(sim):
    _, plugin, xp = sim
    spoken = say(
        plugin, "vertical speed", "set_vertical_speed", {"vs": "1500", "direction": "down"}
    )
    assert spoken == ["Descend 1500 feet per minute"]
    assert xp.commands_fired == ["sim/autopilot/vertical_speed"]
    assert xp.datarefs[VVI_DIAL] == -1500.0


def test_vertical_speed_direction_from_the_selected_altitude(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/cockpit2/autopilot/altitude_dial_ft"] = 9000.0
    xp.datarefs["sim/cockpit2/gauges/indicators/altitude_ft_pilot"] = 5000.0
    assert say(plugin, "vertical speed", "set_vertical_speed", {"vs": "800"}) == [
        "Climb 800 feet per minute, to 9000"  # and the selected altitude is captured
    ]


def test_vertical_speed_without_a_direction_asks(sim):
    _, plugin, xp = sim
    assert say(plugin, "vertical speed", "set_vertical_speed", {"vs": "800"}) == [
        "Climb or descend?"
    ]
    assert xp.writes == []


def test_vertical_speed_on_an_aircraft_probed_without_it(sim):
    _, plugin, xp = sim
    folder = plugin.tel.log_dir / "aircraft"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Other_7__other.json").write_text(
        json.dumps({"steps": {"mode_vertical_speed": "no_change"}})
    )
    aircraft(plugin, xp)
    assert say(
        plugin, "vertical speed", "set_vertical_speed", {"vs": "800", "direction": "up"}
    ) == ["This aircraft has no vertical speed mode"]
    assert xp.commands_fired == []


@pytest.mark.parametrize(
    ("mode", "command", "spoken"),
    [
        ("heading", "sim/autopilot/heading", "Heading on"),
        ("approach", "sim/autopilot/approach", "Approach on"),
        ("altitude hold", "sim/autopilot/altitude_hold", "Altitude hold on"),
        ("level change", "sim/autopilot/level_change", "Level change on"),
        ("v nav", "sim/autopilot/vnav", "V NAV on"),
        ("l nav", "sim/autopilot/gpss", "L NAV on"),
    ],
)
def test_mode_on(sim, mode, command, spoken):
    _, plugin, xp = sim
    assert say(plugin, "mode", "set_autopilot_mode", {"mode": mode}) == [spoken]
    assert xp.commands_fired == [command]


def test_mode_already_on_is_not_pressed_again(sim):
    _, plugin, xp = sim  # pressing would toggle it off
    xp.datarefs[AP + "heading_status"] = 2
    assert say(plugin, "mode", "set_autopilot_mode", {"mode": "heading", "state": "on"}) == [
        "Heading already on"
    ]
    assert xp.commands_fired == []


def test_mode_off_with_unreadable_state_is_refused(sim):
    _, plugin, xp = sim
    del xp.datarefs[AP + "nav_status"]
    assert say(plugin, "mode", "set_autopilot_mode", {"mode": "nav", "state": "off"}) == [
        "Cannot read Nav mode state"
    ]
    assert xp.commands_fired == []


def test_mode_the_aircraft_lacks(sim):
    _, plugin, xp = sim
    xp.missing_commands.add("sim/autopilot/vnav")
    aircraft(plugin, xp)
    assert say(plugin, "mode", "set_autopilot_mode", {"mode": "v nav"}) == [
        "V NAV mode is not available on this aircraft"
    ]


def test_autothrottle_on(sim):
    _, plugin, xp = sim
    assert say(plugin, "auto throttle", "toggle_autothrottle", {"state": "on"}) == [
        "Autothrottle on"
    ]
    assert xp.commands_fired == ["sim/autopilot/autothrottle_on"]


def test_airbus_athr_uses_its_own_button(sim):
    _, plugin, xp = sim
    aircraft(plugin, xp, commands=A333)
    xp.datarefs[AP + "autothrottle_enabled"] = -1  # A/THR off (not even armed)
    say(plugin, "auto throttle", "toggle_autothrottle", {"state": "on"})
    assert xp.commands_fired == ["laminar/A333/autopilot/a_thr_toggle"]


def test_airbus_athr_armed_counts_as_on(sim):
    # A/THR flips -1 (off) and 0 (armed); armed is on.
    _, plugin, xp = sim
    aircraft(plugin, xp, commands=A333)
    xp.datarefs[AP + "autothrottle_enabled"] = 0
    assert say(plugin, "auto throttle", "toggle_autothrottle", {"state": "on"}) == [
        "Autothrottle already on"
    ]


def test_airbus_athr_that_wont_disarm_explains_the_levers(sim, monkeypatch):
    # At FLX/MCT/TOGA the A330's script holds A/THR at armed every frame.
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    aircraft(plugin, xp, commands=A333)
    xp.datarefs[AP + "autothrottle_enabled"] = 0
    say(plugin, "auto throttle", "toggle_autothrottle", {"state": "off"})
    plugin._flight_loop()
    assert xp.spoken[-1] == (
        "Autothrottle did not respond. A/THR stays armed while the thrust levers are at "
        "FLX, MCT or TOGA. Move them to CL."
    )


def test_no_autothrottle_where_the_aircrafts_files_have_none(sim):
    # X-Plane's own autothrottle "worked" in the Baron and King Air probes.
    _, plugin, xp = sim
    aircraft(plugin, xp, engine_type=1)
    plugin._athr_scan = (plugin.profile.info.acf_path, False)
    assert say(plugin, "auto throttle", "toggle_autothrottle", {"state": "on"}) == [
        "This aircraft has no autothrottle"
    ]
    assert xp.commands_fired == []


def test_an_aircrafts_own_athr_button_is_used_even_if_the_scan_found_nothing(sim):
    _, plugin, xp = sim
    aircraft(plugin, xp, commands=A333)
    plugin._athr_scan = (plugin.profile.info.acf_path, False)
    xp.datarefs[AP + "autothrottle_enabled"] = -1
    say(plugin, "auto throttle", "toggle_autothrottle", {"state": "on"})
    assert xp.commands_fired == ["laminar/A333/autopilot/a_thr_toggle"]


def test_the_aircraft_file_scan_runs_off_the_main_thread_and_is_logged(sim, monkeypatch, tmp_path):
    module, plugin, xp = sim
    acf = tmp_path / "Jet" / "jet.acf"
    acf.parent.mkdir()
    acf.write_text("P sim/cockpit2/autopilot/autothrottle_enabled\n")
    plugin._athr_scan = (str(acf), None)
    plugin._scan_aircraft_files(str(acf))
    assert plugin._athr_scan == (str(acf), True)
    assert events(plugin, "aircraft_files")[-1]["autothrottle_shown"] is True


def test_autothrottle_probed_absent(sim):
    _, plugin, xp = sim
    folder = plugin.tel.log_dir / "aircraft"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Other_1__other.json").write_text(
        json.dumps({"steps": {"autothrottle_toggle": "no_change"}})
    )
    aircraft(plugin, xp, engine_type=1)
    assert say(plugin, "auto throttle", "toggle_autothrottle", {"state": "on"}) == [
        "This aircraft has no autothrottle"
    ]


@pytest.mark.parametrize(
    ("pressure", "inhg", "spoken"),
    [
        ("2992", 29.92, "Altimeter two niner niner two"),
        ("29.92", 29.92, "Altimeter two niner niner two"),
        ("1013", 1013 * 0.02953, "Q N H one zero one three"),
    ],
)
def test_altimeter_sets_both_sides(sim, pressure, inhg, spoken):
    _, plugin, xp = sim
    assert say(plugin, "altimeter", "set_altimeter", {"pressure": pressure}) == [spoken]
    assert xp.datarefs[BARO] == pytest.approx(inhg)
    assert xp.datarefs[
        "sim/cockpit2/gauges/actuators/barometer_setting_in_hg_copilot"
    ] == pytest.approx(inhg)


def test_altimeter_standard(sim):
    _, plugin, xp = sim
    assert say(plugin, "set standard", "set_altimeter", {"setting": "standard"}) == [
        "Altimeter standard"
    ]
    assert xp.commands_fired == ["sim/instruments/barometer_std"]


def test_737_standard_presses_its_efis_button(sim):
    _, plugin, xp = sim
    aircraft(plugin, xp, commands=B738)
    say(plugin, "set standard", "set_altimeter", {"setting": "standard"})
    assert xp.commands_fired == ["laminar/B738/EFIS_control/capt/push_button/std_press"]


def test_flaps_by_degrees_on_the_baron(sim):
    _, plugin, xp = sim  # Baron notches are 13 and 30 degrees
    assert say(plugin, "flaps", "set_flaps", {"flap_setting": "15"}) == ["Flaps 15"]
    assert xp.datarefs[FLAP_HANDLE] == 0.5
    assert say(plugin, "flaps", "set_flaps", {"flap_setting": "25"}) == [
        "Flaps 25 is not a setting on this aircraft"
    ]


def test_flaps_by_label_on_the_737(sim):
    _, plugin, xp = sim
    aircraft(
        plugin,
        xp,
        commands=B738,
        **{
            "sim/aircraft/controls/acf_flap_detents": 8,
            "sim/aircraft/controls/acf_flap_dn": [0.0, 3, 5, 8, 10, 12, 14, 18, 40, 0],
            "sim/aircraft/view/acf_Vfe": 162.0,
        },
    )
    say(plugin, "flaps", "set_flaps", {"flap_setting": "15"})
    assert xp.datarefs[FLAP_HANDLE] == pytest.approx(
        5 / 8
    )  # notch 5, though modelled at 12 degrees


def test_flaps_full(sim):
    _, plugin, xp = sim
    assert say(plugin, "flaps", "set_flaps", {"position": "full"}) == ["Flaps full"]
    assert xp.datarefs[FLAP_HANDLE] == 1.0


BARON_FLAP_LIMITS = {"first": 152.0, "full": 122.0}  # its .acf: _Vfe1_kts, _Vfem_kts


def baron_flap_limits(plugin):
    plugin._flap_limits = (plugin.profile.info.acf_path, dict(BARON_FLAP_LIMITS))


def test_flaps_above_flap_speed_need_confirmation_and_can_be_overridden(sim):
    _, plugin, xp = sim
    baron_flap_limits(plugin)
    xp.datarefs["sim/cockpit2/gauges/indicators/airspeed_kts_pilot"] = 140.0
    assert say(plugin, "flaps", "set_flaps", {"flap_setting": "30"}) == [
        "Confirm flaps, above flap speed. Say override, or repeat to confirm."
    ]
    assert xp.datarefs[FLAP_HANDLE] == 0.0
    assert say(plugin, "override") == ["Override", "Flaps 30"]
    assert xp.datarefs[FLAP_HANDLE] == 1.0


def test_flaps_up_and_down_unchanged(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/cockpit2/gauges/indicators/airspeed_kts_pilot"] = 150.0
    say(plugin, "flaps", "set_flaps", {"position": "down"})  # one notch: no new rule
    assert xp.commands_fired == ["sim/flight_controls/flaps_down"]


def test_overlay_shows_the_probe_not_the_last_command(probing):
    # In flight: probe results appeared under the last command's HEARD/MEANT.
    _, plugin, xp = probing
    airborne(xp)
    xp.effects["sim/autopilot/heading"] = toggle_heading
    say(plugin, "autopilot on", "toggle_autopilot", {"state": "on"})
    press_probe(plugin, xp)
    press_probe(plugin, xp)
    settle(plugin, xp)
    lines = plugin.overlay_state.lines()
    assert [ln.badge for ln in lines] == ["PROBE", "WORKS", "NEXT"]
    assert lines[1].text == "heading mode" and lines[2].text.startswith("nav mode")
    assert lines[0].text == "Beechcraft Baron 58  air probe" and lines[0].detail == "1 / 15"
    probe_speech = [e for e in events(plugin, "speak") if e.get("origin") == "probe"]
    assert probe_speech and all("autopilot" not in e["text"].lower() for e in probe_speech)


# ---------------------------------------------------------------------------
# 737 MCP buttons, and the autopilot dropping out during a probe
# ---------------------------------------------------------------------------
B738_MCP = (
    "laminar/B738/autopilot/cmd_a_press",
    "laminar/B738/autopilot/cmd_b_press",
    "laminar/B738/autopilot/hdg_sel_press",
    "laminar/B738/autopilot/vorloc_press",
    "laminar/B738/autopilot/app_press",
    "laminar/B738/autopilot/alt_hld_press",
    "laminar/B738/autopilot/vs_press",
    "laminar/B738/autopilot/lvl_chg_press",
    "laminar/B738/autopilot/vnav_press",
    "laminar/B738/autopilot/lnav_press",
)


def test_737_autopilot_on_presses_cmd_a(sim):
    _, plugin, xp = sim
    profile = load_aircraft(plugin, xp, 7, commands=B738_MCP)
    assert "b738_mcp" in profile.overrides
    say(plugin, "autopilot on", "toggle_autopilot", {"state": "on"})
    assert xp.commands_fired == ["laminar/B738/autopilot/cmd_a_press"]
    say(plugin, "autopilot two on", "toggle_autopilot", {"state": "on", "instance": "2"})
    assert xp.commands_fired[-1] == "laminar/B738/autopilot/cmd_b_press"


def test_737_autopilot_off_still_uses_the_checked_toggle(sim):
    _, plugin, xp = sim
    load_aircraft(plugin, xp, 7, commands=B738_MCP)
    xp.datarefs["sim/cockpit2/autopilot/servos_on"] = 1
    say(plugin, "autopilot off", "toggle_autopilot", {"state": "off"})
    assert xp.commands_fired == ["sim/autopilot/servos_toggle"]


def test_737_probe_presses_its_own_mcp_buttons(probing):
    _, plugin, xp = probing
    load_aircraft(plugin, xp, 7, commands=B738_MCP)
    airborne(xp)
    xp.effects["laminar/B738/autopilot/hdg_sel_press"] = toggle_heading
    press_probe(plugin, xp)
    press_probe(plugin, xp)
    assert settle(plugin, xp) == ["Heading mode: works. Next, nav mode."]
    assert xp.commands_fired == ["laminar/B738/autopilot/hdg_sel_press"] * 2


def test_autopilot_dropping_out_pauses_the_probe_for_a_retry(probing):
    _, plugin, xp = probing
    airborne(xp)

    def heading_and_drop(x):  # the 737 in flight: the servos drop out
        toggle_heading(x)
        x.datarefs[AP + "servos_on"] = 0

    xp.effects["sim/autopilot/heading"] = heading_and_drop
    press_probe(plugin, xp)
    press_probe(plugin, xp)
    assert settle(plugin, xp) == [
        "The autopilot disconnected during heading mode. Re-engage it, then press the probe key to retry."
    ]
    assert plugin._probe is not None and plugin._probe["index"] == 0  # same step next
    assert "mode_heading" not in plugin._probe["results"]
    # Heading was pressed off before the drop; it's pressed again so the
    # retry starts where the run was.
    assert xp.commands_fired == ["sim/autopilot/heading"] * 2
    assert xp.datarefs[AP + "heading_mode"] == 1
    assert [ln.badge for ln in plugin.overlay_state.lines()] == ["PROBE", "PAUSED", "NEXT"]
    # The pilot re-engages; this time the aircraft keeps it.
    xp.datarefs[AP + "servos_on"] = 1
    xp.datarefs[AP + "heading_mode"], xp.datarefs[AP + "heading_status"] = 1, 2
    xp.effects["sim/autopilot/heading"] = toggle_heading
    press_probe(plugin, xp)
    assert settle(plugin, xp) == ["Heading mode: works. Next, nav mode."]


# ---------------------------------------------------------------------------
# Saving every step, carrying on after a stop, and moving between steps
# ---------------------------------------------------------------------------
BARON_FILE = "Beechcraft_Baron_58__Baron_58.json"


def saved(plugin):
    return json.loads((plugin.tel.log_dir / "aircraft" / BARON_FILE).read_text())


def press_stop(plugin, xp):
    before = len(xp.spoken)
    plugin._on_probe_stop_cmd(None, xp.CommandBegin, None)
    return xp.spoken[before:]


def press_move(plugin, xp, delta):
    before = len(xp.spoken)
    handler = plugin._on_probe_skip_cmd if delta > 0 else plugin._on_probe_back_cmd
    handler(None, xp.CommandBegin, None)
    return xp.spoken[before:]


def test_each_step_is_saved_as_soon_as_it_is_measured(probing):
    _, plugin, xp = probing
    airborne(xp)
    xp.effects["sim/autopilot/heading"] = toggle_heading
    press_probe(plugin, xp)
    press_probe(plugin, xp)
    settle(plugin, xp)
    assert plugin._probe is not None  # still running, and already on disk
    record = saved(plugin)
    assert record["steps"] == {"mode_heading": "works"}
    assert record["resume"] == {"air": "mode_nav"} and record["end"] == "in_progress"


def test_a_stopped_run_carries_on_where_it_stopped(probing):
    # In flight the 737's run stopped at ALT HOLD, then LVL CHG, and each new
    # run started again from heading mode.
    _, plugin, xp = probing
    airborne(xp)
    xp.effects["sim/autopilot/heading"] = lambda x: x.datarefs.__setitem__(AP + "heading_mode", 0)
    press_probe(plugin, xp)
    press_probe(plugin, xp)
    settle(plugin, xp, loops=4)  # heading didn't restore: stopped
    assert plugin._probe is None and saved(plugin)["resume"] == {"air": "mode_nav"}
    xp.datarefs[AP + "heading_mode"] = 1  # the pilot puts heading back
    assert press_probe(plugin, xp) == [
        "Probe ready. Continuing at nav mode, step 2 of 15. Press again to run it."
    ]
    assert plugin._probe["index"] == 1
    assert plugin.overlay_state.lines()[0].detail == "1 / 15"


def test_a_complete_run_clears_the_place(probing):
    _, plugin, xp = probing
    xp.datarefs["sim/flightmodel/controls/parkbrake"] = 1.0
    for _ in range(4):
        press_probe(plugin, xp)
        settle(plugin, xp)
    assert xp.spoken[-1] == "Probe complete. Results saved."
    assert saved(plugin)["resume"] == {}
    assert press_probe(plugin, xp)[0].startswith("Probe ready. 3 ground steps.")


def test_stop_twice_starts_over(probing):
    _, plugin, xp = probing
    airborne(xp)
    xp.effects["sim/autopilot/heading"] = toggle_heading
    press_probe(plugin, xp)
    press_probe(plugin, xp)
    settle(plugin, xp)
    assert press_stop(plugin, xp) == [
        "Probe stopped. Results saved. The probe key continues with nav mode."
    ]
    assert press_stop(plugin, xp) == ["Probe reset. The next run starts from the first step."]
    assert saved(plugin)["resume"] == {}
    assert saved(plugin)["steps"] == {"mode_heading": "works"}  # results are kept
    assert press_probe(plugin, xp)[0].startswith("Probe ready. 15 air steps.")
    assert press_stop(plugin, xp)[0].startswith("Probe stopped.")
    assert press_stop(plugin, xp) == ["Probe reset. The next run starts from the first step."]
    assert press_stop(plugin, xp) == ["No probe running"]


def test_stopping_mid_step_finishes_the_step_first(probing):
    _, plugin, xp = probing
    airborne(xp)
    xp.effects["sim/autopilot/heading"] = toggle_heading
    press_probe(plugin, xp)
    press_probe(plugin, xp)  # heading pressed, not yet put back
    assert press_stop(plugin, xp) == ["Stopping after this step"]
    assert settle(plugin, xp) == [
        "Heading mode: works.",
        "Probe stopped. Results saved. The probe key continues with nav mode.",
    ]
    assert xp.datarefs[AP + "heading_mode"] == 1  # put back before stopping


def test_skip_and_back_reach_one_step_without_running_the_rest(probing):
    _, plugin, xp = probing
    airborne(xp)
    assert press_move(plugin, xp, 1) == ["Press the probe key first"]
    press_probe(plugin, xp)
    assert press_move(plugin, xp, 1) == ["Skipped to nav mode, step 2 of 15."]
    assert press_move(plugin, xp, -1) == ["Back to heading mode, step 1 of 15."]
    assert press_move(plugin, xp, -1) == ["Back to altitude dial, step 15 of 15."]  # wraps
    assert xp.commands_fired == []
    lines = plugin.overlay_state.lines()
    assert lines[0].detail == "14 / 15" and lines[-1].text.startswith("altitude dial")


def test_a_run_with_no_modes_on_asks_once_then_runs_anyway(probing):
    # The 737 engaged with neither heading nor altitude hold on: nothing to
    # press to put the pitch mode back, and ALT HOLD didn't restore.
    _, plugin, xp = probing
    airborne(xp)
    xp.datarefs[AP + "heading_mode"], xp.datarefs[AP + "altitude_mode"] = 0, 3
    press_probe(plugin, xp)
    assert press_probe(plugin, xp) == [
        "Heading and altitude hold are not both on, so this step may not restore. "
        "Select them, or press again to run it anyway."
    ]
    assert xp.commands_fired == []
    assert [ln.badge for ln in plugin.overlay_state.lines()] == ["PROBE", "WAIT", "NEXT"]
    press_probe(plugin, xp)
    assert xp.commands_fired == ["sim/autopilot/heading"]


def test_altitude_arm_left_on_is_harmless_and_the_probe_carries_on(probing):
    # The 737: moving the altitude dial armed altitude capture, and putting
    # the dial back left it armed. That stopped the probe on its last step.
    _, plugin, xp = probing
    airborne(xp)

    def heading_and_arm(x):
        toggle_heading(x)
        x.datarefs[AP + "altitude_hold_armed"] = 1  # sticks

    xp.effects["sim/autopilot/heading"] = heading_and_arm
    press_probe(plugin, xp)
    press_probe(plugin, xp)
    assert settle(plugin, xp, loops=5) == [
        "Heading mode: works. Altitude arm stayed on, which is harmless. Next, nav mode."
    ]
    assert xp.commands_fired[-1] == "sim/autopilot/altitude_arm"  # tried to disarm first
    step = events(plugin, "probe_step")[-1]
    assert step["restored"] is False and step["note"].startswith("altitude arm")
    assert [ln.badge for ln in plugin.overlay_state.lines()] == ["PROBE", "WORKS", "NOTE", "NEXT"]


def test_autothrottle_a_mode_turned_on_is_turned_back_off(probing):
    # The 737: LVL CHG engaged the autothrottle, and pressing LVL CHG again
    # left it on.
    _, plugin, xp = probing
    airborne(xp)

    def level_change(x):
        if x.datarefs[AP + "altitude_mode"] == 5:
            x.datarefs[AP + "altitude_mode"] = 6
        else:
            x.datarefs[AP + "altitude_mode"] = 5
            x.datarefs[AP + "autothrottle_enabled"] = 1

    def autothrottle_toggle(x):
        x.datarefs[AP + "autothrottle_enabled"] = 1 - x.datarefs[AP + "autothrottle_enabled"]

    xp.effects["sim/autopilot/level_change"] = level_change
    xp.effects["sim/autopilot/autothrottle_toggle"] = autothrottle_toggle
    press_probe(plugin, xp)
    go_to(plugin, "mode_level_change")
    press_probe(plugin, xp)
    assert settle(plugin, xp, loops=5) == ["Level change: works. Next, V NAV."]
    assert xp.commands_fired == [
        "sim/autopilot/level_change",
        "sim/autopilot/level_change",
        "sim/autopilot/autothrottle_toggle",
    ]
    assert xp.datarefs[AP + "autothrottle_enabled"] == 0


def test_a_pause_puts_back_what_the_step_pressed(probing):
    # The A330: APP was pressed, the autopilot dropped, and APP stayed armed
    # into the next steps, so back course couldn't restore.
    _, plugin, xp = probing
    airborne(xp)
    drops = [True]

    def approach(x):
        on = x.datarefs[AP + "approach_status"] == 1
        x.datarefs[AP + "approach_status"] = x.datarefs[AP + "nav_status"] = 0 if on else 1
        if drops and drops.pop():
            x.datarefs[AP + "servos_on"] = 0

    xp.effects["sim/autopilot/approach"] = approach
    press_probe(plugin, xp)
    go_to(plugin, "mode_approach")
    press_probe(plugin, xp)
    assert settle(plugin, xp)[0].startswith("The autopilot disconnected during approach mode.")
    assert xp.datarefs[AP + "approach_status"] == 0 and xp.datarefs[AP + "nav_status"] == 0
    xp.datarefs[AP + "servos_on"] = 1
    press_probe(plugin, xp)
    assert settle(plugin, xp) == ["Approach mode: works. Next, back course."]


def test_a_pause_that_changed_nothing_presses_nothing_more(probing):
    _, plugin, xp = probing
    airborne(xp)
    xp.effects["sim/autopilot/NAV"] = lambda x: x.datarefs.__setitem__(AP + "servos_on", 0)
    press_probe(plugin, xp)
    go_to(plugin, "mode_nav")
    press_probe(plugin, xp)
    assert settle(plugin, xp)[0].startswith("The autopilot disconnected during nav mode.")
    assert xp.commands_fired == ["sim/autopilot/NAV"]


# ---------------------------------------------------------------------------
# From the flights of 2026-11-07: override said with the command, a bare
# "landing gear", gear up while climbing out
# ---------------------------------------------------------------------------
def climbing_out(xp, agl_ft=600.0, vs_fpm=1200.0):
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs["sim/flightmodel/position/y_agl"] = agl_ft / 3.28084
    xp.datarefs["sim/flightmodel/position/groundspeed"] = 70.0
    xp.datarefs["sim/flightmodel/position/vh_ind_fpm"] = vs_fpm


def test_gear_up_climbing_out_is_not_held(sim):
    _, plugin, xp = sim
    climbing_out(xp)
    assert say(plugin, "landing gear up", "toggle_landing_gear", {"state": "up"}) == ["Gear up"]
    assert xp.commands_fired == [GEAR_UP]


def test_gear_up_low_and_level_is_still_held(sim):
    _, plugin, xp = sim
    climbing_out(xp, vs_fpm=0.0)
    spoken = say(plugin, "landing gear up", "toggle_landing_gear", {"state": "up"})
    assert spoken[0].startswith("Confirm gear up, low altitude")
    assert xp.commands_fired == []


def test_override_said_with_the_held_command(sim):
    _, plugin, xp = sim
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})  # held: on the ground
    # In flight: heard as "override landing get up", conf 0.89, no "gear".
    spoken = say(plugin, "override landing get up", "toggle_landing_gear", {"state": "up"}, 0.89)
    assert spoken == ["Override", "Gear up"]
    assert xp.commands_fired == [GEAR_UP]
    assert events(plugin, "override_with_command")[-1]["intent"] == "toggle_landing_gear"


def test_override_with_a_garbled_rest_still_overrides(sim):
    _, plugin, xp = sim
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})
    assert say(plugin, "override landing get how") == ["Override", "Gear up"]


def test_override_naming_another_command_is_that_command(sim):
    _, plugin, xp = sim
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})
    say(plugin, "override flaps up", "set_flaps", {"position": "up"})
    assert GEAR_UP not in xp.commands_fired
    assert events(plugin, "safety_override") == []


def test_override_contradicting_the_held_command_does_not_confirm_it(sim):
    _, plugin, xp = sim
    say(plugin, "gear", "toggle_landing_gear", {"state": "up"})
    say(plugin, "override gear down", "toggle_landing_gear", {"state": "down"})
    assert GEAR_UP not in xp.commands_fired
    assert events(plugin, "safety_override") == []


def test_override_with_a_command_and_nothing_held_is_just_the_command(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "override landing gear up", "toggle_landing_gear", {"state": "up"})
    assert spoken[0].startswith("Confirm gear up")  # still held on the ground
    assert xp.commands_fired == []


@pytest.mark.parametrize(
    ("text", "question"),
    [
        ("landing gear", "Gear up or down?"),
        ("flaps", "Flaps up or down?"),
        ("parking brake", "Parking brake on or off?"),
    ],
)
def test_a_bare_command_name_asks_which_way(sim, text, question):
    # v17 classed "landing gear" as no command and stayed silent, twice.
    _, plugin, xp = sim
    assert say(plugin, text, "None", {}, 0.999) == [question]
    assert xp.commands_fired == []
    assert events(plugin, "bare_command")[-1]["predicted"] == "None"


def test_a_bare_name_the_model_understood_is_left_to_the_model(sim):
    # "heading" names the autopilot-mode command too; a confident heading
    # reading must not turn into "Which mode?".
    _, plugin, xp = sim
    spoken = say(plugin, "heading", "set_autopilot_heading", {"degrees": "361"})
    assert spoken == ["Invalid heading value"]
    assert events(plugin, "bare_command") == []


# ---------------------------------------------------------------------------
# Part B: FMS modes, altimeter by knob steps, altitude capture
# ---------------------------------------------------------------------------
def probe_file(plugin, steps, key="Other_7__other"):
    folder = plugin.tel.log_dir / "aircraft"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{key}.json").write_text(json.dumps({"steps": steps}))


def test_737_vnav_without_a_flight_plan_says_so(sim, monkeypatch):
    # The probe saw the 737's VNAV button do nothing (no route in the FMC);
    # it's the aircraft's own button, so it's tried, and the failure explained.
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    probe_file(plugin, {"mode_vnav": "no_change"})
    aircraft(plugin, xp, commands=B738_MCP)
    assert say(plugin, "v nav", "set_autopilot_mode", {"mode": "v nav"}) == ["V NAV on"]
    assert xp.commands_fired == ["laminar/B738/autopilot/vnav_press"]
    plugin._flight_loop()
    assert xp.spoken[-1] == "V NAV did not engage. It needs a flight plan in the F M C."


def test_generic_vnav_measured_as_nothing_is_refused(sim):
    # The Baron: sim/autopilot/vnav did nothing, and it has no button for it.
    _, plugin, xp = sim
    probe_file(plugin, {"mode_vnav": "no_change"}, key="Other_1__other")
    aircraft(plugin, xp, engine_type=1)
    assert say(plugin, "v nav", "set_autopilot_mode", {"mode": "v nav"}) == [
        "V NAV mode is not available on this aircraft"
    ]


def test_a330_managed_modes_push_the_fcu_knobs(sim):
    _, plugin, xp = sim
    knobs = (
        "laminar/A333/autopilot/altitude_knob_push",
        "laminar/A333/autopilot/heading_knob_push",
    )
    aircraft(plugin, xp, commands=A333 + knobs)
    say(plugin, "v nav", "set_autopilot_mode", {"mode": "v nav"})
    say(plugin, "l nav", "set_autopilot_mode", {"mode": "l nav"})
    assert xp.commands_fired == [
        "laminar/A333/autopilot/altitude_knob_push",
        "laminar/A333/autopilot/heading_knob_push",
    ]
    # Managed vertical is read from fms_vnav, not the sim's vnav_status.
    assert plugin.profile.status_dataref("mode_vnav", "x") == "sim/cockpit2/autopilot/fms_vnav"


def knob(step):
    def turn(sign):
        def press(x):
            x.datarefs[BARO] = round(x.datarefs[BARO] + sign * step, 4)

        return press

    return turn


@pytest.mark.parametrize(("step", "final"), [(0.01, 29.92), (0.0295, 29.9135)])
def test_altimeter_by_knob_steps_where_writes_dont_hold(sim, step, final):
    # The 737 put a written setting back; its knob (barometer up/down) works.
    _, plugin, xp = sim
    probe_file(plugin, {"dial_baro": "no_change"})
    aircraft(plugin, xp)
    turn = knob(step)
    xp.effects["sim/instruments/barometer_up"] = turn(1)
    xp.effects["sim/instruments/barometer_down"] = turn(-1)
    assert say(plugin, "altimeter", "set_altimeter", {"pressure": "2992"}) == []
    spoken = settle(plugin, xp, loops=40)
    assert spoken == ["Altimeter two niner niner two"]
    assert xp.datarefs[BARO] == pytest.approx(final, abs=1e-4)
    end = events(plugin, "baro_step_end")[-1]
    assert end["ok"] and end["presses"] >= 6 and plugin._baro_step is None
    assert events(plugin, "speak")[-1]["cmd"] == "c9999"  # the reply is the command's


def test_altimeter_knob_that_does_nothing_is_reported(sim):
    _, plugin, xp = sim
    probe_file(plugin, {"dial_baro": "no_change"})
    aircraft(plugin, xp)
    say(plugin, "altimeter", "set_altimeter", {"pressure": "2992"})
    assert settle(plugin, xp, loops=12) == [
        "Altimeter not set: the knob did not move it. It reads three zero one two."
    ]


def climbing_with_vs(xp, actual=5000.0, target=9000.0):
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs[AP + "servos_on"] = 1
    xp.datarefs["sim/cockpit2/autopilot/altitude_dial_ft"] = target
    xp.datarefs["sim/cockpit2/gauges/indicators/altitude_ft_pilot"] = actual

    def vs_mode(x):
        x.datarefs[AP + "altitude_mode"], x.datarefs[AP + "vvi_status"] = 4, 2

    xp.effects["sim/autopilot/vertical_speed"] = vs_mode


def test_vertical_speed_arms_the_aircrafts_own_capture(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "ARM_CAPTURE_DELAY_SEC", 0.0)
    climbing_with_vs(xp)
    say(plugin, "vertical speed", "set_vertical_speed", {"vs": "800", "direction": "up"})
    plugin._flight_loop()
    assert xp.commands_fired == ["sim/autopilot/vertical_speed", "sim/autopilot/altitude_arm"]


def test_capture_already_armed_by_the_aircraft_is_left_alone(sim, monkeypatch):
    # The 737 arms it when V/S engages; the arm command would toggle it off.
    module, plugin, xp = sim
    monkeypatch.setattr(module, "ARM_CAPTURE_DELAY_SEC", 0.0)
    climbing_with_vs(xp)
    xp.datarefs[AP + "altitude_hold_armed"] = 1
    say(plugin, "vertical speed", "set_vertical_speed", {"vs": "800", "direction": "up"})
    plugin._flight_loop()
    assert "sim/autopilot/altitude_arm" not in xp.commands_fired


def test_no_capture_says_how_to_level_off(sim):
    # Option (a): the probe said altitude arm does nothing, Vimaan's capture off.
    _, plugin, xp = sim
    probe_file(plugin, {"mode_altitude_arm": "no_change"})
    aircraft(plugin, xp)
    climbing_with_vs(xp)
    assert say(
        plugin, "vertical speed", "set_vertical_speed", {"vs": "800", "direction": "up"}
    ) == [
        "Climb 800 feet per minute. This aircraft can't capture altitude. Say altitude hold at 9000"
    ]


def test_vimaan_capture_presses_altitude_hold_near_the_target(sim):
    # Option (b), switched on with its key.
    _, plugin, xp = sim
    probe_file(plugin, {"mode_altitude_arm": "no_change"})
    aircraft(plugin, xp)
    climbing_with_vs(xp)
    plugin._on_capture_cmd(None, xp.CommandBegin, None)
    assert xp.spoken[-1] == "Vimaan altitude capture on"
    assert say(
        plugin, "vertical speed", "set_vertical_speed", {"vs": "800", "direction": "up"}
    ) == ["Climb 800 feet per minute. Leveling off at 9000"]
    plugin._flight_loop()
    assert "sim/autopilot/altitude_hold" not in xp.commands_fired  # 4,000 ft to go
    xp.datarefs["sim/cockpit2/gauges/indicators/altitude_ft_pilot"] = 8940.0
    xp.datarefs["sim/flightmodel/position/vh_ind_fpm"] = 800.0  # band = 80 ft
    assert settle(plugin, xp, loops=1) == ["Altitude hold, 9000"]
    assert xp.commands_fired[-1] == "sim/autopilot/altitude_hold"
    assert events(plugin, "capture_end")[-1]["ok"] is True


def test_vimaan_capture_stands_down_when_the_pilot_changes_mode(sim):
    _, plugin, xp = sim
    probe_file(plugin, {"mode_altitude_arm": "no_change"})
    aircraft(plugin, xp)
    climbing_with_vs(xp)
    plugin._capture_enabled = True
    say(plugin, "vertical speed", "set_vertical_speed", {"vs": "800", "direction": "up"})
    xp.datarefs[AP + "altitude_mode"] = 6  # the pilot pressed ALT
    assert settle(plugin, xp, loops=1) == ["Altitude capture cancelled, mode changed"]
    assert plugin._capture is None


def test_cancel_drops_vimaan_capture(sim):
    _, plugin, xp = sim
    probe_file(plugin, {"mode_altitude_arm": "no_change"})
    aircraft(plugin, xp)
    climbing_with_vs(xp)
    plugin._capture_enabled = True
    say(plugin, "vertical speed", "set_vertical_speed", {"vs": "800", "direction": "up"})
    assert say(plugin, "cancel") == ["Cancelled"]
    assert plugin._capture is None
    assert events(plugin, "cancelled")[-1]["dropped"] == ["altitude capture"]


def test_new_altitude_while_climbing_arms_capture(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "ARM_CAPTURE_DELAY_SEC", 0.0)
    climbing_with_vs(xp)
    xp.datarefs[AP + "altitude_mode"] = 4
    xp.datarefs["sim/flightmodel/position/vh_ind_fpm"] = 900.0
    assert say(plugin, "altitude", "set_autopilot_altitude", {"altitude": "7000"}) == [
        "Altitude 7000 feet"
    ]
    plugin._flight_loop()
    assert xp.commands_fired == ["sim/autopilot/altitude_arm"]


# ---------------------------------------------------------------------------
# Automated-test bridge (vimaan_nlu.sim_bridge, Phase A)
# ---------------------------------------------------------------------------
@pytest.fixture
def bridged(sim, monkeypatch, tmp_path):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "BRIDGE_POLL_SEC", 0.0)
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    root = tmp_path / "Vimaan_Logs" / "test"
    (root / "inbox").mkdir(parents=True)
    (root / "bridge.on").write_text("")
    plugin._bridge_next = 0.0  # the fixture's first loop scheduled the next look
    return module, plugin, xp, root


_REQ = iter(range(10**6))


def ask(plugin, root, loops=8, **req):
    """Send one request through the inbox; run the flight loop until answered."""
    rid = req.setdefault("id", f"r{next(_REQ)}")
    (root / "inbox" / f"{rid}.json").write_text(json.dumps(req))
    out = root / "outbox" / f"{rid}.json"
    for _ in range(loops):
        plugin._flight_loop()
        if out.exists():
            return json.loads(out.read_text())
    return None


def test_the_bridge_is_off_unless_switched_on(sim, tmp_path):
    _, plugin, xp = sim
    inbox = tmp_path / "Vimaan_Logs" / "test" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "r1.json").write_text(json.dumps({"id": "r1", "op": "state"}))
    plugin._bridge_next = 0.0
    plugin._flight_loop()
    assert (inbox / "r1.json").exists()  # nobody read it
    assert "Test bridge on" not in xp.spoken


def test_switching_the_bridge_on_and_off_is_announced(bridged):
    _, plugin, xp, root = bridged
    state = ask(plugin, root, op="state")
    assert xp.spoken[-1] == "Test bridge on"
    assert state["ok"] and state["ready"] and state["aircraft"]["icao"] == "BE58"
    assert state["autopilot"]["servos_on"] == 0 and state["on_ground"] == 1
    (root / "bridge.on").unlink()
    plugin._flight_loop()
    assert xp.spoken[-1] == "Test bridge off"
    assert [e["on"] for e in events(plugin, "test_bridge")] == [True, False]


def test_a_said_command_comes_back_with_what_happened(bridged):
    _, plugin, xp, root = bridged
    NLU["autopilot on"] = ("toggle_autopilot", {"state": "on"}, 0.99)
    xp.effects["sim/autopilot/servos_toggle"] = lambda x: x.datarefs.update(
        {"sim/cockpit2/autopilot/servos_on": 1}
    )
    r = ask(plugin, root, op="say", text="autopilot on")
    assert r["ok"] and r["outcome"] == "executed" and r["heard"] == "autopilot on"
    assert r["intent"] == "toggle_autopilot" and r["replies"] == ["Autopilot 1 on"]
    assert {"command": "sim/autopilot/servos_toggle", "ok": True} in r["actions"]
    assert r["checks"] and all(c["ok"] for c in r["checks"])  # waited for the read-back
    press = events(plugin, "ptt_press")[-1]
    assert press["source"] == "test" and press["text"] == "autopilot on"


def test_a_recording_goes_through_recognition(bridged, tmp_path):
    import wave

    _, plugin, xp, root = bridged
    wav = tmp_path / "gear.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(struct.pack("<2h", 8000, -8000) * 8000)
    STT["next"] = "gear down"
    NLU["gear down"] = ("toggle_landing_gear", {"state": "down"}, 0.99)
    r = ask(plugin, root, op="say", wav=str(wav))
    assert r["heard"] == "gear down" and r["intent"] == "toggle_landing_gear"
    capture = [e for e in events(plugin, "stage") if e.get("stage") == "capture"][-1]
    assert capture["source_file"] == str(wav)  # saving a copy: test_voice_worker


def test_a_said_command_waits_for_the_selector_it_turns(bridged):
    # The 737's transponder: the result comes back with the reply after the
    # last click, not when the command returned "pending".
    _, plugin, xp, root = bridged
    aircraft(plugin, xp, commands=B738_XPDR)
    xp.datarefs[B738_XPDR_POS] = 1
    turn = selector(xp, B738_XPDR_POS, *B738_XPDR[:2], 0, 5, delay=2)
    NLU["transponder alt"] = ("set_transponder_mode", {"xpdr_mode": "alt"}, 0.99)
    rid = "sel"
    (root / "inbox" / f"{rid}.json").write_text(
        json.dumps({"id": rid, "op": "say", "text": "transponder alt"})
    )
    out = root / "outbox" / f"{rid}.json"
    for _ in range(30):
        turn()
        plugin._flight_loop()
        if out.exists():
            break
    r = json.loads(out.read_text())
    assert r["outcome"] == "pending" and r["replies"] == ["Transponder altitude"]
    assert xp.datarefs[B738_XPDR_POS] == 3


def test_one_said_command_at_a_time(bridged):
    _, plugin, xp, root = bridged
    NLU["gear up"] = ("toggle_landing_gear", {"state": "up"}, 0.99)
    (root / "inbox" / "a.json").write_text(json.dumps({"id": "a", "op": "say", "text": "gear up"}))
    (root / "inbox" / "b.json").write_text(json.dumps({"id": "b", "op": "say", "text": "gear up"}))
    plugin._flight_loop()
    busy = json.loads((root / "outbox" / "b.json").read_text())
    assert busy == {"id": "b", "ok": False, "error": "busy"}


def test_read_write_and_commands(bridged):
    _, plugin, xp, root = bridged
    alt = "sim/cockpit2/autopilot/altitude_dial_ft"
    r = ask(
        plugin,
        root,
        op="read",
        datarefs=[
            "sim/cockpit2/autopilot/servos_on",
            "sim/flightmodel/engine/ENGN_running[1]",
            "sim/aircraft/view/acf_ICAO",
            "sim/nope",
        ],
    )
    assert r["values"] == {
        "sim/cockpit2/autopilot/servos_on": 0,
        "sim/flightmodel/engine/ENGN_running[1]": 0,
        "sim/aircraft/view/acf_ICAO": "BE58",
        "sim/nope": None,
    }
    assert (
        len(
            ask(plugin, root, op="read", datarefs=["sim/flightmodel/engine/ENGN_running"])[
                "values"
            ]["sim/flightmodel/engine/ENGN_running"]
        )
        == 16
    )
    assert ask(plugin, root, op="write", dataref=alt, value=7000) == {
        "id": "r" + str(int(events(plugin, "test_request")[-1]["request"][1:])),
        "ok": True,
        "readback": 7000.0,
    }
    xp.read_only.add(alt)
    assert "read-only" in ask(plugin, root, op="write", dataref=alt, value=1)["error"]
    assert ask(plugin, root, op="command", name="sim/autopilot/no_such_command")["ok"] is False
    assert ask(plugin, root, op="command", name="sim/autopilot/heading")["ok"] is True
    cmd = events(plugin, "sim_command")[-1]
    assert (
        cmd["origin"] == "test"
        and cmd["cmd"] is None
        and xp.commands_fired[-1] == "sim/autopilot/heading"
    )
    assert "unknown op" in ask(plugin, root, op="fly")["error"]


def test_placing_the_aircraft_converts_units(bridged):
    _, plugin, xp, root = bridged
    r = ask(
        plugin,
        root,
        op="place",
        latitude=47.5,
        longitude=-122.3,
        elevation_ft=6000,
        heading_true=180,
        speed_kt=150,
    )
    assert r["ok"]
    kind, lat, lon, elev_m, hdg, mps = xp.placed[-1]
    assert (kind, lat, lon, hdg) == ("location", 47.5, -122.3, 180.0)
    assert elev_m == pytest.approx(1828.8) and mps == pytest.approx(77.17, abs=0.01)
    assert ask(plugin, root, op="airport", icao="KSEA")["ok"] and xp.placed[-1] == (
        "airport",
        "KSEA",
    )


def test_a_disconnect_through_the_bridge_is_diagnosed_like_the_pilots(bridged, monkeypatch):
    # So an automated test can check the disconnect callout (Phase D).
    module, plugin, xp, root = bridged
    monkeypatch.setattr(module, "STATE_WATCH_SEC", 0.0)
    xp.datarefs[SERVOS] = xp.datarefs[AP_ON] = 1
    plugin._next_state_watch = 0.0
    plugin._flight_loop()
    xp.effects["sim/autopilot/servos_fdir_off"] = drop
    assert ask(plugin, root, op="command", name="sim/autopilot/servos_fdir_off")["ok"]
    plugin._flight_loop()
    (e,) = events(plugin, "autopilot_disconnect")
    assert e["cause"] == "command" and e["commands"][0]["by_vimaan"] is False
    assert "Autopilot disconnected by a switch or button." in xp.spoken


# ---------------------------------------------------------------------------
# Fixes from the A2 runs (2026-11-12)
# ---------------------------------------------------------------------------
def test_full_flaps_are_held_above_the_full_flap_limit_not_the_first_notchs(sim):
    # The 737 took full flaps at 240 kt: acf_Vfe (250) is its first notch's limit.
    _, plugin, xp = sim
    baron_flap_limits(plugin)
    xp.datarefs["sim/cockpit2/gauges/indicators/airspeed_kts_pilot"] = 140.0  # 122 < 140 < 152
    assert say(plugin, "flaps", "set_flaps", {"position": "full"})[0].startswith("Confirm flaps")
    hold = events(plugin, "safety_hold")[-1]
    assert hold["rule"] == "flaps_above_vfe" and hold["context"]["vfe_kt"] == 122.0
    say(plugin, "cancel")
    # The first notch (approach, 13 degrees) is allowed to 152 kt.
    assert say(plugin, "flaps", "set_flaps", {"flap_setting": "15"}) == ["Flaps 15"]
    assert xp.datarefs[FLAP_HANDLE] == pytest.approx(0.5)


def test_flap_limits_are_read_from_the_acf_on_load(sim, tmp_path):
    _, plugin, xp = sim
    acf = tmp_path / "Baron_58.acf"
    acf.write_bytes(
        b"I\n1200 version\nP acf/_Vfe1_kts 152.000000000\nP acf/_Vfem_kts 122.000000000\n"
    )
    plugin._athr_scan = (str(acf), None)
    plugin._scan_aircraft_files(str(acf))
    assert plugin._flap_limits == (str(acf), BARON_FLAP_LIMITS)
    assert events(plugin, "aircraft_files")[-1]["flap_limits"] == BARON_FLAP_LIMITS


def test_without_the_acf_limits_the_published_one_is_used(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/cockpit2/gauges/indicators/airspeed_kts_pilot"] = 160.0  # above acf_Vfe 152
    assert say(plugin, "flaps", "set_flaps", {"position": "full"})[0].startswith("Confirm flaps")


def test_an_aircrafts_overrides_go_with_it(sim):
    # A330 loaded after the 737: the 737's commands stay registered, so its
    # overrides applied and Vimaan pressed the 737's CMD A on the A330.
    _, plugin, xp = sim
    assert "b738_mcp" in load_aircraft(plugin, xp, 7, commands=B738_MCP).overrides
    del xp.datarefs["laminar/B738/autopilot/cmd_a_status"]  # the 737 unloads; commands stay
    xp.datarefs["sim/aircraft/view/acf_ICAO"] = "A333"
    profile = load_aircraft(plugin, xp, 7, commands=A333)
    assert profile.overrides == ("airbus_fcu",)
    xp.datarefs["sim/cockpit2/autopilot/servos_on"] = 0
    say(plugin, "autopilot on", "toggle_autopilot", {"state": "on"})
    assert "laminar/B738/autopilot/cmd_a_press" not in xp.commands_fired


def test_overrides_are_looked_for_again_after_a_load(sim):
    _, plugin, xp = sim
    profile = load_aircraft(plugin, xp, 7, commands=B738_MCP, live=())
    assert "b738_mcp" not in profile.overrides  # its scripts' datarefs not there yet
    xp.datarefs["laminar/B738/autopilot/cmd_a_status"] = 0
    plugin._recheck_overrides(plugin._aircraft)
    assert "b738_mcp" in plugin.profile.overrides
    assert events(plugin, "aircraft")[-1]["reason"] == "overrides_changed"
    plugin._recheck_overrides(("other", "/x.acf"))  # another aircraft by now: nothing
    assert len(events(plugin, "aircraft")) == 3


def test_a_pilot_disconnect_right_after_autopilot_on_is_called_out(flying):
    # 2026-11-12: Vimaan's "autopilot on" 2.5 s earlier was taken as the cause.
    _, plugin, xp = flying
    plugin._last_action = {
        "cmd": "c1",
        "intent": "toggle_autopilot",
        "state": "on",
        "t": time.perf_counter(),
    }
    xp.effects["sim/autopilot/servos_fdir_off"] = drop
    xp.commandOnce("sim/autopilot/servos_fdir_off")
    before = len(xp.spoken)
    plugin._flight_loop()
    assert xp.spoken[before:] == ["Autopilot disconnected by a switch or button."]
    assert events(plugin, "autopilot_disconnect")[-1]["cause"] == "command"


# ---------------------------------------------------------------------------
# From the A2 re-run (2026-11-12): the 737's A/T ARM switch; overspeed
# ---------------------------------------------------------------------------
ATHR = "sim/cockpit2/autopilot/autothrottle_enabled"
ARM = "sim/autopilot/autothrottle_arm"  # the 737's A/T ARM switch (its cockpit object)


def b738_with_athr_switch(plugin, xp, armed):
    """The 737's autothrottle: -1 with the ARM switch off, 0 armed, 1 holding speed."""
    xp.datarefs[ATHR] = 0 if armed else -1

    def arm(x):
        x.datarefs[ATHR] = max(x.datarefs[ATHR], 0)

    def engage(x):  # engages only once armed
        if x.datarefs[ATHR] >= 0:
            x.datarefs[ATHR] = 1

    xp.effects[ARM] = arm
    xp.effects["sim/autopilot/autothrottle_on"] = engage
    return load_aircraft(plugin, xp, 7, commands=B738_MCP)


def test_a_hard_off_autothrottle_is_armed_first(sim, monkeypatch):
    # Re-runs: the 737's switch was off, the A/T stayed at -1: "did not respond".
    module, plugin, xp = sim
    monkeypatch.setattr(module, "ATHR_ARM_SETTLE_SEC", 0.0)
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    b738_with_athr_switch(plugin, xp, armed=False)
    assert say(plugin, "auto throttle on", "toggle_autothrottle", {"state": "on"}) == [
        "Arming the autothrottle"
    ]
    assert xp.commands_fired == [ARM]
    before = len(xp.spoken)
    plugin._flight_loop()  # the deferred engage, then the read-backs
    plugin._flight_loop()
    assert xp.commands_fired == [ARM, "sim/autopilot/autothrottle_on"]
    assert xp.spoken[before:] == ["Autothrottle on"] and xp.datarefs[ATHR] == 1
    assert all(e["ok"] for e in events(plugin, "verify"))


def test_an_armed_autothrottle_engages_straight_away(sim):
    _, plugin, xp = sim
    b738_with_athr_switch(plugin, xp, armed=True)
    assert say(plugin, "auto throttle on", "toggle_autothrottle", {"state": "on"}) == [
        "Autothrottle on"
    ]
    assert xp.commands_fired == ["sim/autopilot/autothrottle_on"]


def test_autothrottle_off_leaves_it_armed(sim):
    _, plugin, xp = sim
    b738_with_athr_switch(plugin, xp, armed=True)
    xp.datarefs[ATHR] = 1
    say(plugin, "auto throttle off", "toggle_autothrottle", {"state": "off"})
    assert ARM not in xp.commands_fired
    assert "sim/autopilot/autothrottle_hard_off" not in xp.commands_fired


def test_the_a330s_armed_a_thr_is_not_armed_again(sim):
    # A/THR counts armed (0) as on; its own toggle, never the generic arm.
    _, plugin, xp = sim
    aircraft(plugin, xp, commands=A333, **{ATHR: -1})
    say(plugin, "auto throttle on", "toggle_autothrottle", {"state": "on"})
    assert ARM not in xp.commands_fired


def test_an_overspeed_disconnect_is_called_out(flying):
    # A330 at 335 kt (Vmo 330): its high-speed protection took the autopilot.
    _, plugin, xp = flying
    xp.datarefs["sim/cockpit2/gauges/indicators/airspeed_kts_pilot"] = 240.0  # Baron Vne 223
    drop(xp)
    before = len(xp.spoken)
    plugin._flight_loop()
    assert xp.spoken[before:] == ["Autopilot disconnected. Overspeed."]
    e = events(plugin, "autopilot_disconnect")[-1]
    assert e["cause"] == "overspeed" and e["overspeed"] == "240 kt, limit 223"


# ---------------------------------------------------------------------------
# From the fourth run (2026-11-12): altitude hold armed is not on
# ---------------------------------------------------------------------------
ALT_HOLD = "sim/cockpit2/autopilot/altitude_hold_status"


def test_altitude_hold_armed_is_pressed_to_hold_now(sim, monkeypatch):
    # After level change the 737 and A330 had altitude hold armed (1), and
    # "altitude hold" said "already on" while they kept climbing.
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    xp.datarefs[ALT_HOLD] = 1
    xp.effects["sim/autopilot/altitude_hold"] = lambda x: x.datarefs.update({ALT_HOLD: 2})
    assert say(plugin, "altitude hold", "set_autopilot_mode", {"mode": "altitude hold"}) == [
        "Altitude hold on"
    ]
    assert xp.commands_fired == ["sim/autopilot/altitude_hold"]
    plugin._flight_loop()
    assert events(plugin, "verify")[-1]["ok"] is True


def test_altitude_hold_holding_is_already_on(sim):
    _, plugin, xp = sim
    xp.datarefs[ALT_HOLD] = 2
    assert say(plugin, "altitude hold", "set_autopilot_mode", {"mode": "altitude hold"}) == [
        "Altitude hold already on"
    ]
    assert xp.commands_fired == []


def test_altitude_hold_off_while_only_armed_presses_nothing(sim):
    # A press would engage it, the opposite of what was asked.
    _, plugin, xp = sim
    xp.datarefs[ALT_HOLD] = 1
    assert say(
        plugin, "altitude hold off", "set_autopilot_mode", {"mode": "altitude hold", "state": "off"}
    ) == ["Altitude hold already off"]
    assert xp.commands_fired == []


def test_approach_armed_still_counts_as_on(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/cockpit2/autopilot/approach_status"] = 1
    assert say(plugin, "approach", "set_autopilot_mode", {"mode": "approach"}) == [
        "Approach already on"
    ]


# ---------------------------------------------------------------------------
# Tier 2: radios and transponder
# ---------------------------------------------------------------------------
RADIO = "sim/cockpit2/radios/actuators/"
XPDR_MODE = RADIO + "transponder_mode"


def swaps(active, standby):
    def press(x):
        x.datarefs[active], x.datarefs[standby] = x.datarefs[standby], x.datarefs[active]

    return press


def test_com_standby_written_in_khz(sim):
    _, plugin, xp = sim
    spoken = say(
        plugin, "com 2 standby 121.9", "set_com_standby", {"com_port": "2", "frequency": "121.9"}
    )
    assert xp.datarefs[RADIO + "com2_standby_frequency_hz_833"] == 121900
    assert xp.datarefs[RADIO + "com2_frequency_hz_833"] == 122800  # active untouched
    assert spoken == ["COM 2 standby one two one decimal niner"]


def test_com_flip_reads_back_the_new_active(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    xp.effects["sim/radios/com1_standy_flip"] = swaps(
        RADIO + "com1_frequency_hz_833", RADIO + "com1_standby_frequency_hz_833"
    )
    spoken = say(plugin, "flip com 1", "swap_com_frequency", {"com_port": "1"})
    assert spoken == ["COM 1 active one two one decimal five"]
    assert xp.commands_fired == ["sim/radios/com1_standy_flip"]
    plugin._flight_loop()
    assert events(plugin, "verify")[-1]["ok"] is True


def test_com_flip_the_radio_ignores_is_reported(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    say(plugin, "flip com", "swap_com_frequency", {})
    assert settle(plugin, xp, loops=1) == ["COM 1 active did not respond"]


def test_nav_frequency_written_in_10_khz(sim):
    _, plugin, xp = sim
    spoken = say(
        plugin,
        "tune nav 2 to 113.9",
        "set_nav_frequency",
        {"nav_port": "2", "nav_frequency": "113.9"},
    )
    assert xp.datarefs[RADIO + "nav2_frequency_hz"] == 11390
    assert spoken == ["NAV 2 set to one one three decimal niner"]


@pytest.mark.parametrize("frequency", ["121.5", "107.9", "118.0"])
def test_nav_frequency_outside_the_nav_band_is_not_written(sim, frequency):
    _, plugin, xp = sim
    spoken = say(plugin, "nav 1 " + frequency, "set_nav_frequency", {"nav_frequency": frequency})
    assert xp.writes == [] and spoken == ["Frequency out of range"]


def test_nav_standby_then_flip(sim):
    _, plugin, xp = sim
    xp.effects["sim/radios/nav1_standy_flip"] = swaps(
        RADIO + "nav1_frequency_hz", RADIO + "nav1_standby_frequency_hz"
    )
    assert say(
        plugin,
        "nav 1 standby 109.5",
        "set_nav_standby",
        {"nav_port": "1", "nav_frequency": "109.5"},
    ) == ["NAV 1 standby one zero niner decimal five"]
    assert say(plugin, "swap nav 1", "swap_nav_frequency", {"nav_port": "1"}) == [
        "NAV 1 active one zero niner decimal five"
    ]
    assert xp.datarefs[RADIO + "nav1_frequency_hz"] == 10950


def test_nav_port_3_rejected(sim):
    _, plugin, xp = sim
    spoken = say(plugin, "swap nav 3", "swap_nav_frequency", {"nav_port": "3"})
    assert xp.commands_fired == [] and "not supported" in spoken[-1]


def test_course_sets_the_obs(sim):
    _, plugin, xp = sim
    assert say(plugin, "course 090", "set_course", {"degrees": "090"}) == ["Course zero niner zero"]
    assert xp.datarefs[RADIO + "nav1_obs_deg_mag_pilot"] == 90.0
    assert say(plugin, "nav 2 course 180", "set_course", {"degrees": "180", "nav_port": "2"}) == [
        "NAV 2 course one eight zero"
    ]
    assert xp.datarefs[RADIO + "nav2_obs_deg_mag_pilot"] == 180.0


def test_squawk_keeps_leading_zeros(sim):
    _, plugin, xp = sim
    assert say(plugin, "squawk 0120", "set_transponder_code", {"squawk": "0120"}) == [
        "Squawk zero one two zero"
    ]
    assert xp.datarefs[RADIO + "transponder_code"] == 120


@pytest.mark.parametrize("code", ["1280", "123", "12345"])
def test_squawk_that_is_not_four_octal_digits_is_refused(sim, code):
    _, plugin, xp = sim
    spoken = say(plugin, "squawk " + code, "set_transponder_code", {"squawk": code})
    assert xp.writes == [] and spoken == ["Invalid squawk. Four digits, zero to seven."]


def test_emergency_squawk_needs_the_pilot_to_confirm(sim):
    _, plugin, xp = sim
    assert say(plugin, "squawk 7700", "set_transponder_code", {"squawk": "7700"}) == [
        "Confirm emergency squawk. Say override, or repeat to confirm."
    ]
    assert xp.writes == []
    # A different code is not a confirmation of this one.
    say(plugin, "squawk 7600", "set_transponder_code", {"squawk": "7600"})
    assert xp.writes == []
    assert say(plugin, "squawk 7600", "set_transponder_code", {"squawk": "7600"}) == [
        "Squawk seven six zero zero"
    ]
    assert xp.datarefs[RADIO + "transponder_code"] == 7600


def test_transponder_mode_with_x_planes_commands(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    xp.effects["sim/transponder/transponder_alt"] = lambda x: x.datarefs.update({XPDR_MODE: 3})
    assert say(plugin, "transponder alt", "set_transponder_mode", {"xpdr_mode": "alt"}) == [
        "Transponder altitude"
    ]
    assert xp.commands_fired == ["sim/transponder/transponder_alt"]
    plugin._flight_loop()
    assert events(plugin, "verify")[-1]["ok"] is True


@pytest.mark.parametrize(("mode", "said"), [(3, "on"), (6, "on"), (7, "alt"), (1, "standby")])
def test_transponder_already_there_presses_nothing(sim, mode, said):
    # "On" means replying: it never drops altitude reporting to mode A.
    _, plugin, xp = sim
    xp.datarefs[XPDR_MODE] = mode
    spoken = say(plugin, "transponder " + said, "set_transponder_mode", {"xpdr_mode": said})
    assert spoken[0].startswith("Transponder already") and xp.commands_fired == []


def test_transponder_mode_missing_is_asked(sim):
    _, plugin, _ = sim
    assert say(plugin, "transponder", "set_transponder_mode", {}) == [
        "Transponder off, standby, on, or altitude?"
    ]


def test_ident(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
    assert say(plugin, "ident", "set_transponder_mode", {"xpdr_mode": "ident"}) == [
        "Transponder is in standby. Say transponder on first."
    ]
    xp.datarefs[XPDR_MODE] = 3
    xp.effects["sim/transponder/transponder_ident"] = lambda x: x.datarefs.update(
        {RADIO.replace("actuators", "indicators") + "transponder_id": 1}
    )
    assert say(plugin, "ident", "set_transponder_mode", {"xpdr_mode": "ident"}) == ["Ident"]
    assert xp.commands_fired == ["sim/transponder/transponder_ident"]
    plugin._flight_loop()
    assert events(plugin, "verify")[-1]["ok"] is True


B738_XPDR_POS = "laminar/B738/knob/transponder_pos"
B738_XPDR = (
    "laminar/B738/knob/transponder_mode_up",
    "laminar/B738/knob/transponder_mode_dn",
    "laminar/B738/push_button/transponder_ident_dn",
)


def selector(xp, position, up, down, low, high, delay=0):
    """A selector one click moves, ``delay`` frames after the click."""
    queued = []

    def click(sign):
        def press(x):
            queued.append([delay, sign])

        return press

    def apply():
        for q in list(queued):
            if q[0] <= 0:
                queued.remove(q)
                xp.datarefs[position] = max(low, min(high, xp.datarefs[position] + q[1]))
            else:
                q[0] -= 1

    xp.effects[up], xp.effects[down] = click(1), click(-1)
    return apply


def run_selectors(plugin, xp, *applies, loops=20):
    before = len(xp.spoken)
    for _ in range(loops):
        for apply in applies:
            apply()
        plugin._flight_loop()
    return xp.spoken[before:]


def test_737_transponder_turns_its_selector(sim):
    _, plugin, xp = sim
    assert "b738_transponder" in aircraft(plugin, xp, commands=B738_XPDR).overrides
    xp.datarefs[B738_XPDR_POS] = 1  # STBY
    turn = selector(xp, B738_XPDR_POS, *B738_XPDR[:2], 0, 5)
    assert say(plugin, "transponder alt", "set_transponder_mode", {"xpdr_mode": "alt"}) == []
    assert run_selectors(plugin, xp, turn) == ["Transponder altitude"]
    assert xp.datarefs[B738_XPDR_POS] == 3  # XPNDR
    assert xp.commands_fired == [B738_XPDR[0]] * 2 and plugin._knob_step is None
    assert events(plugin, "speak")[-1]["cmd"] == "c9999"
    # XPNDR, TA and TA/RA all reply: "on" leaves it there.
    assert say(plugin, "transponder on", "set_transponder_mode", {"xpdr_mode": "on"}) == [
        "Transponder already on"
    ]
    assert say(plugin, "transponder off", "set_transponder_mode", {"xpdr_mode": "off"}) == [
        "This transponder has no off. Say standby."
    ]
    say(plugin, "transponder standby", "set_transponder_mode", {"xpdr_mode": "standby"})
    assert run_selectors(plugin, xp, turn) == ["Transponder standby"]
    assert xp.datarefs[B738_XPDR_POS] == 1


def test_selector_that_moves_a_frame_late_is_not_overshot(sim):
    _, plugin, xp = sim
    aircraft(plugin, xp, commands=B738_XPDR)
    xp.datarefs[B738_XPDR_POS] = 1
    turn = selector(xp, B738_XPDR_POS, *B738_XPDR[:2], 0, 5, delay=2)
    say(plugin, "transponder alt", "set_transponder_mode", {"xpdr_mode": "alt"})
    assert run_selectors(plugin, xp, turn) == ["Transponder altitude"]
    assert xp.datarefs[B738_XPDR_POS] == 3 and len(xp.commands_fired) == 2


def test_selector_that_never_moves_is_reported(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module, "KNOB_STUCK_FRAMES", 3)
    aircraft(plugin, xp, commands=B738_XPDR)
    xp.datarefs[B738_XPDR_POS] = 1
    say(plugin, "transponder alt", "set_transponder_mode", {"xpdr_mode": "alt"})
    assert run_selectors(plugin, xp, loops=6) == ["Transponder not set: the selector did not move."]
    assert events(plugin, "knob_step_end")[-1]["ok"] is False


def test_737_ident_uses_its_own_button(sim):
    _, plugin, xp = sim
    aircraft(plugin, xp, commands=B738_XPDR)
    xp.datarefs[XPDR_MODE] = 2
    assert say(plugin, "ident", "set_transponder_mode", {"xpdr_mode": "ident"}) == ["Ident"]
    assert xp.commands_fired == ["laminar/B738/push_button/transponder_ident_dn"]


A333_XPDR = "laminar/A333/transponder/"
A333_XPDR_COMMANDS = tuple(
    A333_XPDR + n
    for n in (
        "auto_on_off_left",
        "auto_on_off_right",
        "alt_rpt_off",
        "alt_rpt_on",
        "ta_ra_left",
        "ta_ra_right",
    )
)


def test_a330_transponder_turns_its_three_selectors(sim):
    _, plugin, xp = sim
    profile = aircraft(
        plugin,
        xp,
        commands=A333_XPDR_COMMANDS,
        **{A333_XPDR + "alt_rpt_knob_pos": 0, A333_XPDR + "ta_ra_knob_pos": 0},
    )
    assert "a333_transponder" in profile.overrides
    xp.datarefs[A333_XPDR + "auto_on_knob_pos"] = -1  # STBY
    c = A333_XPDR_COMMANDS
    turns = (
        selector(xp, A333_XPDR + "auto_on_knob_pos", c[1], c[0], -1, 1),
        selector(xp, A333_XPDR + "alt_rpt_knob_pos", c[3], c[2], 0, 1),
        selector(xp, A333_XPDR + "ta_ra_knob_pos", c[5], c[4], 0, 2),
    )
    say(plugin, "transponder alt", "set_transponder_mode", {"xpdr_mode": "alt"})
    assert run_selectors(plugin, xp, *turns) == ["Transponder altitude"]
    # AUTO, altitude reporting on, TCAS TA: the aircraft's normal setting.
    knobs = [xp.datarefs[f"{A333_XPDR}{n}_knob_pos"] for n in ("auto_on", "alt_rpt", "ta_ra")]
    assert knobs == [0, 1, 1]
    assert xp.commands_fired == [c[1], c[3], c[5]]
    assert say(plugin, "transponder off", "set_transponder_mode", {"xpdr_mode": "off"}) == [
        "This transponder has no off. Say standby."
    ]
