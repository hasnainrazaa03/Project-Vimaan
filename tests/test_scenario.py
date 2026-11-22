"""The scenario runner (simtest/scenario.py) and the Tier 1 scenarios, with
a stand-in bridge: no X-Plane."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simtest import scenario, tier1  # noqa: E402
from simtest.scenario import Runner, Step, equals, heading_near, levelled_at  # noqa: E402


class FakeBridge:
    def __init__(self, events_path, readings=None, voice_hears=None, voice_outcome="executed"):
        self.events_path = events_path
        self.readings = readings or {}  # name -> value, or list of values (one per read)
        self.voice_hears = voice_hears or {}  # phrase -> what the voice is heard as
        self.voice_outcome = voice_outcome  # the outcome of a say_wav
        self.texts, self.wavs, self.commands, self.writes = [], [], [], []

    def _result(self, text, heard=None, outcome="executed"):
        intent = {"autopilot on": "toggle_autopilot", "autopilot of": "toggle_autopilot"}.get(
            text, "x"
        )
        return {
            "ok": True,
            "heard": heard or text,
            "intent": intent,
            "slots": {"t": text},
            "outcome": outcome,
            "replies": [f"Did {text}"],
            "checks": [{"ok": True}],
            "latency_ms": 5,
        }

    def say_wav(self, path):
        phrase = Path(path).stem
        self.wavs.append(phrase)
        heard = self.voice_hears.get(phrase, phrase)
        return self._result(heard, heard, self.voice_outcome)

    def say_text(self, text):
        self.texts.append(text)
        return self._result(text)

    def read(self, *names):
        out = {}
        for n in names:
            v = self.readings.get(n)
            out[n] = (
                v.pop(0)
                if isinstance(v, list) and len(v) > 1
                else (v[0] if isinstance(v, list) else v)
            )
        return out

    def command(self, name):
        self.commands.append(name)
        return {"ok": True}

    def write(self, name, value):
        self.writes.append((name, value))
        self.readings[name] = value
        return {"ok": True, "readback": value}

    def state(self):
        return {"events": str(self.events_path)}


@pytest.fixture
def fake_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(
        scenario.voice_mod, "synthesize", lambda text, voice: tmp_path / f"{text}.wav"
    )
    monkeypatch.setattr(scenario.time, "sleep", lambda s: None)


def run(bridge, steps, **kw):
    return Runner(bridge, log=lambda line: None, **kw).run(steps)


def test_a_step_passes_on_outcome_reply_and_sim(tmp_path, fake_voice):
    b = FakeBridge(tmp_path / "e.jsonl", {"sim/a": 1})
    (r,) = run(b, [Step("x", say="autopilot on", reply="did", sim=(equals("a on", "sim/a", 1),))])
    assert r["ok"] and r["heard_ok"] and r["understood_ok"] and b.texts == []
    assert r["sim"] == [{"what": "a on", "expected": "1", "actual": 1, "ok": True}]


def test_failures_are_named(tmp_path, fake_voice):
    b = FakeBridge(tmp_path / "e.jsonl", {"sim/a": 0})
    (r,) = run(
        b,
        [
            Step(
                "x",
                say="autopilot on",
                outcomes=("no_op",),
                reply="nope",
                sim=(equals("a on", "sim/a", 1),),
            )
        ],
    )
    assert not r["ok"]
    assert r["problems"] == [
        "outcome executed, wanted no_op",
        "no reply with “nope”",
        "a on: got 0, wanted 1",
    ]


def test_a_sim_check_waits_for_the_aircraft(tmp_path, fake_voice, monkeypatch):
    clock = iter(range(1000))
    monkeypatch.setattr(scenario.time, "monotonic", lambda: next(clock))
    hdg = "sim/cockpit2/gauges/indicators/heading_AHARS_deg_mag_pilot"
    b = FakeBridge(tmp_path / "e.jsonl", {hdg: [180, 220, 262, 268]})
    (r,) = run(b, [Step("turn", say="x", sim=(heading_near(270, tol=15, within_sec=100),))])
    assert r["ok"] and r["sim"][0]["actual"] == 8.0
    alt, vs = (
        "sim/cockpit2/gauges/indicators/altitude_ft_pilot",
        "sim/cockpit2/gauges/indicators/vvi_fpm_pilot",
    )
    b = FakeBridge(tmp_path / "e.jsonl", {alt: [6000, 5000], vs: [-1500, -1400]})
    (r,) = run(b, [Step("level", say="x", sim=(levelled_at(4000, within_sec=2),))])
    assert not r["ok"] and "levelled off at 4000 ft" in r["problems"][0]


def test_a_misunderstood_voice_is_redone_as_text(tmp_path, fake_voice):
    b = FakeBridge(
        tmp_path / "e.jsonl",
        voice_hears={"autopilot on": "autopilot of"},
        voice_outcome="not_understood",
    )
    (r,) = run(b, [Step("x", say="autopilot on")], understood=lambda res, ph: False)
    assert r["heard_ok"] is False and r["understood_ok"] is False and b.texts == ["autopilot on"]
    assert r["misheard"]["heard"] == "autopilot of" and r["ok"]
    assert "redone as text" in scenario.line(r)


@pytest.mark.parametrize("outcome", ["executed", "pending"])
def test_a_misheard_phrase_already_acted_on_is_not_done_twice(tmp_path, fake_voice, outcome):
    # "turn left twenty degrees" heard as "down left twenty degrees": the
    # model's slots differ (direction=down), but the turn was flown. Again as
    # text, it would be flown twice.
    b = FakeBridge(
        tmp_path / "e.jsonl",
        voice_hears={"turn left twenty degrees": "down left twenty degrees"},
        voice_outcome=outcome,
    )
    (r,) = run(b, [Step("x", say="turn left twenty degrees")], understood=lambda res, ph: False)
    assert b.texts == [] and b.wavs == ["turn left twenty degrees"]
    assert r["understood_ok"] is False and r["acted_as_heard"] is True
    assert r["misheard"]["heard"] == "down left twenty degrees"
    assert r["replies"] == ["Did down left twenty degrees"]
    assert "acted on as heard" in scenario.line(r)
    assert scenario.summary([r])["understood_by_voice"] == 0


def test_acted_reads_the_outcome_or_the_actions():
    assert scenario.acted({"outcome": "executed"}) and scenario.acted({"outcome": "pending"})
    assert scenario.acted({"outcome": "x", "actions": [{"command": "c"}]})
    for outcome in ("ambiguous", "rejected", "implausible", "not_understood", None):
        assert not scenario.acted({"outcome": outcome, "actions": []})


def test_heard_differently_but_understood_the_same_is_not_redone(tmp_path, fake_voice):
    b = FakeBridge(tmp_path / "e.jsonl", voice_hears={"altimeter two": "altimeters two"})
    (r,) = run(b, [Step("x", say="altimeter two")], understood=lambda res, ph: True)
    assert r["heard_ok"] is False and r["understood_ok"] is True and b.texts == []


def test_text_mode_skips_recognition(tmp_path, fake_voice):
    b = FakeBridge(tmp_path / "e.jsonl")
    (r,) = run(b, [Step("x", say="gear up")], voice=None)
    assert b.texts == ["gear up"] and b.wavs == [] and r["heard_ok"] is None


def test_callouts_are_read_from_the_session_log(tmp_path, fake_voice):
    events = tmp_path / "e.jsonl"
    before = (datetime.now() - timedelta(minutes=5)).isoformat(timespec="milliseconds")
    after = (datetime.now() + timedelta(seconds=1)).isoformat(timespec="milliseconds")
    events.write_text(
        "\n".join(
            json.dumps(e)
            for e in (
                {"ts": before, "kind": "speak", "text": "Autopilot disconnected. Stick moved."},
                {
                    "ts": after,
                    "kind": "speak",
                    "text": "Autopilot disconnected by a switch or button.",
                },
            )
        )
        + "\n"
    )
    b = FakeBridge(events)
    (ok,) = run(
        b, [Step("callout", command="sim/autopilot/servos_fdir_off", callout="switch or button")]
    )
    (bad,) = run(b, [Step("quiet", say="autopilot off", no_callout="Autopilot disconnected")])
    assert ok["ok"] and b.commands == ["sim/autopilot/servos_fdir_off"]
    assert (
        not bad["ok"] and "Vimaan said" in bad["problems"][0]
    )  # the newer one counts, not the old


def test_summary_and_report(tmp_path):
    results = [
        {"step": "a", "ok": True, "heard_ok": True, "understood_ok": True, "problems": []},
        {"step": "b", "ok": False, "heard_ok": False, "understood_ok": True, "problems": ["x"]},
        {"step": "c", "ok": True, "heard_ok": None, "problems": []},
    ]
    s = scenario.summary(results)
    assert s == {
        "steps": 3,
        "skipped": 0,
        "passed": 2,
        "failed": ["b"],
        "heard_exactly": 1,
        "understood_by_voice": 2,
        "said_by_voice": 2,
    }
    path = scenario.write_report(tmp_path, {"started": "2026-11-12T01:30:00", "aircraft": "baron"})
    assert path.name == "20261112T013000_baron.json"
    assert "✕ b: x" == scenario.line(results[1]).split("  [")[0]


@pytest.mark.parametrize("aircraft", list(tier1.SETUPS))
def test_tier1_scenarios_are_complete(aircraft):
    steps = tier1.steps(aircraft)
    names = [s.name for s in steps]
    assert len(names) == len(set(names)), "step names are unique (the report keys on them)"
    assert all(sum(map(bool, (s.say, s.command, s.write))) == 1 for s in steps)
    said = " | ".join(s.say for s in steps if s.say)
    setup = tier1.SETUPS[aircraft]
    for must in (
        "autopilot on",
        f"descend at {setup.vs_words} feet per minute",
        "frequency one three three decimal one on com two",
        "altimeter standard",
        "flaps full",
    ):
        assert must in said
    assert (f"mach {setup.mach_words}" in said) == setup.jet
    # Without an autothrottle, "auto throttle on" is only said to be refused.
    if setup.autothrottle == "none":
        assert [s.outcomes for s in steps if s.say == "auto throttle on"] == [("rejected",)]


def test_the_fleet_and_the_new_aircraft_are_all_set_up():
    assert set(tier1.FLEET) | set(tier1.NEW) == set(tier1.SETUPS)
    assert not set(tier1.FLEET) & set(tier1.NEW)
    from simtest import xplane

    assert set(tier1.SETUPS) <= set(xplane.AIRCRAFT)


def test_steps_can_depend_on_the_sim_and_on_each_other(tmp_path, fake_voice):
    b = FakeBridge(tmp_path / "e.jsonl", {"sim/fast": 0})
    fast = equals("fast", "sim/fast", 1)
    results = run(
        b,
        [
            Step("held", say="flaps full", only_if=fast),
            Step("cancel", say="cancel", after="held"),
            Step("done", say="flaps full"),
            Step("up", say="flaps up", after="done"),
        ],
    )
    assert [(r["step"], bool(r.get("skipped"))) for r in results] == [
        ("held", True),
        ("cancel", True),
        ("done", False),
        ("up", False),
    ]
    assert scenario.summary(results)["skipped"] == 2


def test_a_pilot_write_step(tmp_path, fake_voice):
    b = FakeBridge(tmp_path / "e.jsonl")
    throttle = "sim/cockpit2/engine/actuators/throttle_ratio_all"
    (r,) = run(b, [Step("levers", write=(throttle, 0.53), sim=(equals("CL", throttle, 0.53),))])
    assert r["ok"] and b.writes == [(throttle, 0.53)]


@pytest.mark.parametrize("aircraft", ["737", "a330"])
def test_jets_manage_thrust_straight_after_autopilot_on(aircraft):
    names = [s.name for s in tier1.steps(aircraft)]
    start = names.index("autopilot on")
    thrust = (
        ["thrust levers to CL (pilot)", "autothrottle on"]
        if aircraft == "a330"
        else ["autothrottle on"]
    )
    assert names[start + 1 : start + 1 + len(thrust)] == thrust
    assert names.index("autothrottle on") < names.index("speed 250") < names.index("mach .78")
    assert "autothrottle on again" in names


# --- Questions (Tier 4): the question key, the answer against the sim --------


class AskingBridge(FakeBridge):
    """Answers on the question key; ``acts`` lists the sim actions a phrase makes."""

    def __init__(self, events_path, readings, replies, acts=None):
        super().__init__(events_path, readings)
        self.replies, self.acts, self.asked = replies, acts or {}, []

    def say_text(self, text, ask=False):
        self.texts.append(text)
        self.asked.append(ask)
        return {
            **self._result(text),
            "outcome": "answered",
            "replies": [self.replies[text]],
            "actions": self.acts.get(text, []),
        }


ALTITUDE = scenario.Answer(
    "altitude",
    ("sim/alt",),
    lambda reply, r: reply == f"Altitude {round(r['sim/alt'], -1):.0f} feet",
    lambda r: f"Altitude {r['sim/alt']}",
)


def test_a_question_is_asked_on_the_question_key_and_checked_against_the_sim(tmp_path):
    b = AskingBridge(
        tmp_path / "e.jsonl", {"sim/alt": [6004.0, 6004.0]}, {"alt?": "Altitude 6000 feet"}
    )
    step = Step("altitude", say="alt?", ask=True, outcomes=("answered",), answer=ALTITUDE)
    (r,) = run(b, [step], voice=None)
    assert r["ok"] and b.asked == [True]
    assert r["answer"]["readings"] == {"sim/alt": 6004.0} and r["actions"] == []


def test_a_wrong_answer_or_a_question_that_moves_the_sim_fails(tmp_path):
    b = AskingBridge(
        tmp_path / "e.jsonl",
        {"sim/alt": 7404.0},
        {"alt?": "Altitude 6000 feet", "gear?": "Gear up"},
        acts={"gear?": [{"command": "sim/flight_controls/landing_gear_down", "ok": True}]},
    )
    steps = [
        Step("altitude", say="alt?", ask=True, outcomes=("answered",), answer=ALTITUDE),
        Step("gear", say="gear?", ask=True, outcomes=("answered",), no_actions=True),
    ]
    wrong, moved = run(b, steps, voice=None)
    assert not wrong["ok"] and "sim shows Altitude 7404.0" in wrong["problems"][0]
    assert not moved["ok"] and "changed the sim" in moved["problems"][0]


def test_the_talk_key_sends_no_ask(tmp_path, fake_voice):
    # A bridge from before the question key still works for every other step.
    b = FakeBridge(tmp_path / "e.jsonl")
    (r,) = run(b, [Step("x", say="autopilot on", reply="did")], voice=None)
    assert r["ok"] and b.texts == ["autopilot on"]
