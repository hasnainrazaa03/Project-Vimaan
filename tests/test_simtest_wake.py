"""Phase L.5.3: the wake word through the test bridge ("hear") and the
simtest scenario that uses it (simtest/wake.py)."""

from __future__ import annotations

import json
import struct
import sys
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vimaan_nlu import sim_bridge  # noqa: E402

from simtest import __main__ as cli  # noqa: E402
from simtest import record, scenario, tier1, wake  # noqa: E402


def test_the_command_in_a_wake_phrase():
    assert scenario.wake_command("hey copilot gear down") == "gear down"
    assert scenario.wake_command("hey copilot ... altitude six thousand") == "altitude six thousand"
    assert scenario.wake_command("hey copilot") == ""
    assert scenario.wake_command("copilot, check the flaps for me") == ""
    assert scenario.wake_command("autopilot on, heading mode") == ""


@pytest.mark.parametrize("aircraft", tier1.FLEET)
def test_the_scenario_on_each_aircraft(aircraft):
    steps = wake.steps(aircraft)
    names = [s.name for s in steps]
    assert len(names) == len(set(names))
    heard = [s for s in steps if s.wake]
    assert len(heard) == 12 and not any(s.wake for s in tier1.hands_off(aircraft))
    talk = [s for s in heard if s.outcomes == ("no_wake",)]
    assert len(talk) == 4 and all(not scenario.wake_command(s.say) for s in talk)
    commands = [s for s in heard if scenario.wake_command(s.say)]
    assert sum(" ... " in s.say for s in commands) == 3  # after a pause
    assert sum(" ... " not in s.say for s in commands) == 4  # one breath (and the question)


def test_record_wake_all():
    words = [p for p, _ in record.phrases("wake", list(tier1.FLEET))]
    assert len(words) == len(set(words)) == 21
    assert "hey copilot ... heading one eight zero" in words and "hey copilot" in words


def test_the_wake_scenario_needs_the_wake_word():
    assert "wake word is off" in cli._model_lacks("wake", {"wake_word": False})
    assert cli._model_lacks("wake", {"wake_word": True}) is None


def test_hear_needs_a_wav(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    assert sim_bridge.problem({"id": "r1", "op": "hear", "wav": str(wav)}) is None
    assert "absolute path" in sim_bridge.problem({"id": "r1", "op": "hear", "wav": "a.wav"})
    assert "no such file" in sim_bridge.problem(
        {"id": "r1", "op": "hear", "wav": str(tmp_path / "b.wav")}
    )


class HearBridge:
    """The runner's side: what it sent, and the plugin's answer."""

    def __init__(self, answer):
        self.answer, self.heard = answer, []

    def hear_wav(self, wav):
        self.heard.append(Path(wav).name)
        return dict(self.answer)


def test_the_runner_says_a_wake_step_on_the_open_mic(monkeypatch, tmp_path):
    monkeypatch.setattr(scenario.voice_mod, "synthesize", lambda text, v: tmp_path / "gen.wav")
    step = scenario.Step("x", say="hey copilot gear down", wake=True)
    result = {"ok": True, "outcome": "executed", "heard": "gear down", "replies": ["Gear down"]}
    bridge = HearBridge(result)
    r = scenario.Runner(bridge, voice=None, log=lambda _: None).run_step(step)
    assert bridge.heard == ["gen.wav"] and r["ok"] and r["heard_ok"]
    # Your recording, when there is one.
    rec = tmp_path / "007.wav"
    bridge = HearBridge(result)
    runner = scenario.Runner(bridge, voice=None, recordings={step.say: rec}, log=lambda _: None)
    runner.run_step(step)
    assert bridge.heard == ["007.wav"]
    # Nothing woke: "no_wake", which talk without the wake word wants.
    talk = scenario.Step(
        "t", say="copilot, check the flaps for me", wake=True, outcomes=("no_wake",)
    )
    r = scenario.Runner(HearBridge({"ok": True, "woke": False}), voice=None).run_step(talk)
    assert r["ok"] and r["outcome"] == "no_wake" and r.get("heard_ok") is None
    # Heard wrong: never redone as text (hearing it is the test).
    bridge = HearBridge({**result, "heard": "gear up", "outcome": "safety_hold"})
    r = scenario.Runner(bridge, voice=None, log=lambda _: None).run_step(step)
    assert not r["ok"] and r["heard_ok"] is False and len(bridge.heard) == 1


def test_a_pause_is_a_second_of_quiet_in_generated_speech(monkeypatch, tmp_path):
    from simtest import voice

    ran = []
    monkeypatch.setattr(voice, "available", lambda: True)
    monkeypatch.setattr(voice.subprocess, "run", lambda cmd, **kw: ran.append(cmd))
    voice.synthesize("hey copilot ... gear down", "Daniel", cache_dir=tmp_path)
    assert ran[-1][-1] == "hey copilot [[slnc 1000]] gear down"


# ---- the plugin's side of "hear" ------------------------------------------
@pytest.fixture
def sim(monkeypatch, tmp_path):
    import test_plugin

    test_plugin.NLU.clear()
    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "BRIDGE_POLL_SEC", 0.0)
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)  # read-backs at once
    plugin._on_worker_message(
        {"type": "ready", "load_ms": 1.0, "model_path": "/fake/v28", "mic": {}, "wake": True}
    )
    plugin._speaking_until = 0.0
    flag, _, outbox = sim_bridge.paths(plugin.tel.log_dir)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("")
    plugin._bridge_next = 0.0
    plugin._service_bridge()  # the bridge on ("Test bridge on" said)
    plugin._speaking_until = 0.0  # ...and over
    wav = tmp_path / "hey.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(struct.pack("<h", 0) * 1600)
    yield module, plugin, xp, outbox, wav
    plugin.XPluginStop()


def answer(outbox, rid):
    path = outbox / f"{rid}.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_a_wake_heard_through_the_bridge_answers_as_a_say(sim):
    _, plugin, xp, outbox, wav = sim
    assert plugin._bridge_request({"id": "r1", "op": "hear", "wav": str(wav)}) is None
    plugin._bridge_next = 0.0
    plugin._service_bridge()  # Vimaan is quiet: played at once
    assert plugin.worker.sent[-1] == {"op": "hear", "path": str(wav)}
    assert plugin._bridge_state()["busy"] and plugin._bridge_state()["wake_word"]
    plugin._on_worker_message({"type": "wake", "wake": 1, "conf": 1.0, "one_breath": True})
    cmd = plugin._active["cmd"]
    plugin._on_worker_message({"type": "captured", "cmd": cmd, "audio_sec": 1.0, "wake": True})
    plugin._on_worker_message(
        {
            "type": "result",
            "cmd": cmd,
            "text": "landing lights on",
            "stt_ms": 3.0,
            "nlu_ms": 2.0,
            "intent": "toggle_lights",
            "confidence": 0.99,
            "slots": {"light": "landing", "switch": "on"},
        }
    )
    for _ in range(300):  # the switch stepped, read back, then the answer
        plugin._bridge_next = 0.0
        plugin._flight_loop()
        if answer(outbox, "r1"):
            break
    said = answer(outbox, "r1")
    assert said["cmd"] == cmd and said["outcome"] in ("executed", "pending")
    assert "Landing lights on" in said["replies"]


def test_nothing_woken_and_a_wake_dropped(sim, monkeypatch):
    module, plugin, _, outbox, wav = sim
    monkeypatch.setattr(module, "BRIDGE_HEAR_WAIT_SEC", 0.0)
    plugin._bridge_request({"id": "r2", "op": "hear", "wav": str(wav)})
    for _ in range(2):  # played, then nothing woke
        plugin._bridge_next = 0.0
        plugin._service_bridge()
    assert answer(outbox, "r2") == {"id": "r2", "ok": True, "woke": False}
    monkeypatch.setattr(module, "BRIDGE_HEAR_WAIT_SEC", 30.0)
    plugin._bridge_request({"id": "r3", "op": "hear", "wav": str(wav)})
    plugin._bridge_next = 0.0
    plugin._service_bridge()
    plugin._speaking_until = 1e12  # Vimaan speaking: the wake is dropped
    plugin._on_worker_message({"type": "wake", "wake": 2, "conf": 1.0, "one_breath": True})
    assert answer(outbox, "r3") == {
        "id": "r3",
        "ok": True,
        "woke": True,
        "taken": False,
        "why": "speaking",
    }


def test_hear_waits_for_vimaan_to_be_quiet(sim):
    # As a pilot would (L.5.3: the next phrase straight after a read-back was
    # dropped, "speaking").
    _, plugin, _, outbox, wav = sim
    plugin._speaking_until = 1e12
    plugin._bridge_request({"id": "r5", "op": "hear", "wav": str(wav)})
    plugin._bridge_next = 0.0
    plugin._service_bridge()
    assert not [m for m in plugin.worker.sent if m.get("op") == "hear"]
    # A wake from the room meanwhile isn't the bridge's.
    plugin._speaking_until = 0.0
    plugin._on_worker_message({"type": "wake", "wake": 9, "conf": 1.0, "one_breath": True})
    assert plugin._bridge_heard["id"] == "r5" and answer(outbox, "r5") is None
    plugin._active = None
    plugin._bridge_next = 0.0
    plugin._service_bridge()
    assert plugin.worker.sent[-1] == {"op": "hear", "path": str(wav)}


def test_hear_refused_with_the_wake_word_off(sim):
    _, plugin, xp, _, wav = sim
    plugin._on_wake_word_cmd(None, xp.CommandBegin, None)  # off
    result = plugin._bridge_request({"id": "r4", "op": "hear", "wav": str(wav)})
    assert result == {"ok": False, "error": "the wake word is off"}
