"""Phase L.5: the wake word in the plugin. A wake from the worker is taken as
a command, as a key pressed, and goes through the same rules; or dropped when
Vimaan is busy or speaking. The worker side is tests/test_wake_word.py."""

import time

import pytest
import test_plugin
from test_plugin import events

LANDING = "sim/cockpit2/switches/landing_lights_on"
LIGHTS_ON = {"light": "landing", "switch": "on"}


@pytest.fixture
def sim(monkeypatch, tmp_path):
    test_plugin.NLU.clear()
    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    plugin._speaking_until = 0.0  # what it said at startup
    yield module, plugin, xp
    plugin.XPluginStop()


def ready(plugin, wake=True, why=None):
    """The worker says it is ready, and whether it can listen for the wake word."""
    msg = {"type": "ready", "load_ms": 1.0, "model_path": "/fake/v28", "mic": {}, "wake": wake}
    msg.update(wake_phrase="hey copilot", **({"wake_off": why} if why else {}))
    plugin._on_worker_message(msg)
    plugin._speaking_until = 0.0


def sent(plugin, op):
    return [m for m in plugin.worker.sent if m.get("op") == op]


def wake(plugin, n=1, one_breath=True):
    plugin._on_worker_message({"type": "wake", "wake": n, "conf": 1.0, "one_breath": one_breath})


def result(plugin, cmd, text="gear down", intent="toggle_landing_gear", slots=None):
    """The worker's reply for a wake's command: what it heard and read."""
    plugin._on_worker_message({"type": "captured", "cmd": cmd, "audio_sec": 1.1, "wake": True})
    plugin._on_worker_message(
        {
            "type": "result",
            "cmd": cmd,
            "text": text,
            "stt_ms": 4.0,
            "nlu_ms": 3.0,
            "normalized": text,
            "intent": intent,
            "confidence": 0.99,
            "slots": slots if slots is not None else {"state": "down"},
            "to_reading": None,
            "two_commands": None,
            "ask": False,
            "heard": None,
            "question": None,
        }
    )


def test_on_by_default_once_the_worker_can(sim):
    _, plugin, _ = sim
    assert sent(plugin, "wake") == []  # the fake worker's own ready: no wake word
    ready(plugin)
    assert sent(plugin, "wake") == [{"op": "wake", "on": True}]
    setting = events(plugin, "wake_word_setting")[-1]
    assert setting["on"] and setting["available"] and setting["phrase"] == "hey copilot"


def test_off_with_its_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("VIMAAN_WAKE_WORD", "0")
    _, plugin, _ = test_plugin._load_plugin(monkeypatch, tmp_path)
    try:
        ready(plugin)
        assert sent(plugin, "wake") == [{"op": "wake", "on": False}]
    finally:
        plugin.XPluginStop()


def test_a_wake_is_a_command_with_the_usual_rules(sim):
    _, plugin, xp = sim
    ready(plugin)
    wake(plugin)
    active = plugin._active
    assert active["source"] == "wake" and active["t_release"] is None
    assert sent(plugin, "wake_take") == [{"op": "wake_take", "wake": 1, "cmd": active["cmd"]}]
    assert len(xp.played) == 1 and xp.played[0][1] == 16000  # the chime
    press = events(plugin, "ptt_press")[-1]
    assert press["source"] == "wake" and press["cmd"] == active["cmd"]
    result(plugin, active["cmd"], "landing lights on", "toggle_lights", LIGHTS_ON)
    test_plugin.run_selectors(plugin, xp)  # the switch, then its read-back
    assert plugin._active is None
    assert xp.datarefs[LANDING] == 1 and "Landing lights on" in xp.spoken
    done = events(plugin, "command_done")[-1]
    assert done["cmd"] == active["cmd"] and done["outcome"] in ("executed", "pending")
    assert events(plugin, "ptt_release")[-1]["source"] == "wake"
    taken = events(plugin, "wake_word")[-1]
    assert taken["taken"] is True and taken["why"] is None


def test_a_safety_hold_applies_to_a_wake_too(sim):
    _, plugin, xp = sim
    ready(plugin)
    wake(plugin)
    cmd = plugin._active["cmd"]
    result(plugin, cmd, "gear up", slots={"state": "up"})  # on the ground: held
    assert "sim/flight_controls/landing_gear_up" not in xp.commands_fired
    assert xp.spoken[-1].endswith("Say override, or repeat to confirm.")


def test_dropped_when_busy_or_speaking(sim):
    _, plugin, _ = sim
    ready(plugin)
    plugin.OnPressCallback(None)  # the pilot is talking on the key
    wake(plugin)
    assert sent(plugin, "wake_drop") == [{"op": "wake_drop", "wake": 1}]
    assert events(plugin, "wake_word")[-1]["why"] == "busy"
    plugin._active = None
    plugin._speaking_until = time.perf_counter() + 10.0
    wake(plugin, n=2)
    assert events(plugin, "wake_word")[-1] == {**events(plugin, "wake_word")[-1], "why": "speaking"}
    assert sent(plugin, "wake_drop")[-1]["wake"] == 2


def test_the_key_wins_over_a_wake_still_listening(sim):
    _, plugin, xp = sim
    ready(plugin)
    wake(plugin, one_breath=False)
    woke = plugin._active["cmd"]
    spoken = len(xp.spoken)
    plugin.OnPressCallback(None)
    assert plugin._active["source"] == "hotkey" and plugin._active["cmd"] != woke
    assert xp.spoken[spoken:] == []  # given up quietly
    done = [e for e in events(plugin, "command_done") if e["cmd"] == woke]
    assert done[-1]["outcome"] == "wake_cancelled"


def test_nothing_said_after_the_wake_ends_quietly(sim):
    _, plugin, xp = sim
    ready(plugin)
    wake(plugin, one_breath=False)
    cmd = plugin._active["cmd"]
    spoken = len(xp.spoken)
    plugin._on_worker_message({"type": "error", "cmd": cmd, "outcome": "no_command"})
    assert plugin._active is None and xp.spoken[spoken:] == []
    assert events(plugin, "command_done")[-1]["outcome"] == "no_command"


def test_a_wake_with_no_reply_times_out(sim, monkeypatch):
    module, plugin, xp = sim
    ready(plugin)
    monkeypatch.setattr(module, "WAKE_LISTEN_MAX_SEC", 0.0)
    wake(plugin, one_breath=False)
    plugin._flight_loop()
    assert plugin._active is None
    assert events(plugin, "command_done")[-1]["outcome"] == "timeout"


def test_vimaan_speaking_mutes_it(sim):
    _, plugin, _ = sim
    plugin._say("Gear down")
    assert sent(plugin, "wake_mute") == []  # not before the worker can listen
    ready(plugin)
    plugin.worker.sent.clear()
    plugin._say("Gear down")
    mute = sent(plugin, "wake_mute")
    assert len(mute) == 1 and 1.0 < mute[0]["sec"] < 3.0


def test_its_command_switches_it_off_and_on(sim):
    _, plugin, xp = sim
    xp_begin = test_plugin.make_xp().CommandBegin
    plugin._on_wake_word_cmd(None, xp_begin, None)
    assert xp.spoken[-1] == "The wake word needs the offline recogniser"
    ready(plugin)
    plugin._on_wake_word_cmd(None, xp_begin, None)
    assert xp.spoken[-1] == "Wake word off" and sent(plugin, "wake")[-1]["on"] is False
    wake(plugin)
    assert events(plugin, "wake_word")[-1]["why"] == "off"
    plugin._on_wake_word_cmd(None, xp_begin, None)
    assert xp.spoken[-1] == "Wake word on" and sent(plugin, "wake")[-1]["on"] is True
    assert "vimaan/toggle_wake_word" in xp.custom_commands
