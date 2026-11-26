"""Phase L.4.6: the checklist scenarios (simtest/checklists.py) and the
runner following a checklist (scenario.Follow), end to end against the
real plugin on the fake sim."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vimaan_nlu.checklists import CHECKLISTS, NAMES  # noqa: E402

from simtest import __main__ as cli  # noqa: E402
from simtest import checklists, flights, scenario, tier1  # noqa: E402
from simtest.bridge import Bridge  # noqa: E402


def test_the_tail_waits_for_a_whole_line(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"kind": "old"}\n')
    tail = scenario.Tail(path, path.stat().st_size)
    assert tail.read() == []
    with open(path, "ab") as fh:
        fh.write(b'{"kind": "a"}\n{"kind": "b", "text": "Gear do')
    assert tail.read() == [{"kind": "a"}]
    with open(path, "ab") as fh:
        fh.write(b'wn"}\nnot json\n')
    assert tail.read() == [{"kind": "b", "text": "Gear down"}]
    assert scenario.Tail(tmp_path / "missing", 0).read() == []


def _follows(steps):
    return [s.follow for s in steps if s.follow is not None]


@pytest.mark.parametrize("aircraft", tier1.FLEET)
def test_the_scenarios_name_real_checklists_and_items(aircraft):
    for steps in (checklists.ground(aircraft), checklists.air(aircraft)):
        names = [s.name for s in steps]
        assert len(names) == len(set(names))  # each step its own line in the report
        for f in _follows(steps):
            items = {i.name for i in CHECKLISTS[f.name]}
            assert f.name in NAMES
            assert set(f.answers) <= items and set(f.sets) <= items and set(f.held) <= items
            for how in f.answers.values():
                assert callable(how) or how in scenario.ANSWERS
    ground = _follows(checklists.ground(aircraft))
    assert {f.name for f in ground} == set(NAMES)  # all six on the runway
    assert any(f.stop for f in ground) and any(f.held for f in ground)
    assert {"cancel", "wait"} <= {a for f in ground for a in f.answers.values()}


def test_again_follows_a_run_of_the_same_checklist():
    for steps in (checklists.ground("baron"), checklists.air("737")):
        seen = set()
        for f in _follows(steps):
            assert not f.again or f.name in seen
            seen.add(f.name)


class _Reads:
    def __init__(self, **values):
        self.values = values

    def read(self, *refs):
        return {r: self.values.get(r) for r in refs}


@pytest.mark.parametrize(
    ("ias", "vle", "answer"),
    [(140, 152, "confirm"), (145, 152, "cancel"), (140, None, "cancel"), (None, 152, "cancel")],
)
def test_the_gear_comes_down_only_below_its_speed(ias, vle, answer):
    bridge = _Reads(**{checklists.IAS: ias, checklists.VLE: vle})
    assert checklists.gear_below_its_speed(bridge) == answer


def test_the_checklists_need_v28():
    lacks = cli._model_lacks("checklists-ground", {"model_intents": ["None", "turn_by"]})
    assert lacks and "v28" in lacks
    assert cli._model_lacks("checklists-air", {"model_intents": ["run_checklist"]}) is None
    assert "checklists-ground" in flights.ON_THE_RUNWAY
    assert flights.FLIGHTS["checklists-ground"]("baron", None)[0].name == "parked on the runway"


# ---- end to end: the runner, the bridge and the real plugin on the fake sim ----
def _run(monkeypatch, tmp_path, steps, setup=None):
    import test_plugin

    test_plugin.NLU.clear()
    for name in NAMES:
        test_plugin.NLU[f"{name} checklist"] = ("run_checklist", {"checklist": name}, 0.99)
    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    results: list = []
    try:
        monkeypatch.setattr(module, "BRIDGE_POLL_SEC", 0.0)
        monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)
        # A short pause after each line, as in the sim (there 0.9 s and more):
        # a "cancel" straight after the start comes before the first item.
        monkeypatch.setattr(module.checklists, "GAP_SEC", 0.25)
        monkeypatch.setattr(module.checklists, "WORD_SEC", 0.0)
        monkeypatch.setattr(module.checklists, "OFFER_WAIT_SEC", 0.2)
        plugin._speaking_until = 0.0
        if setup is not None:
            setup(plugin, xp)
        br = Bridge(tmp_path / "Vimaan_Logs", poll_sec=0.002)
        br.enable()
        plugin._bridge_next = 0.0
        runner = scenario.Runner(
            br, voice=None, poll_sec=0.005, follow_poll_sec=0.005, log=lambda _line: None
        )
        runner.summary_wait_sec = 2.0
        t = threading.Thread(target=lambda: results.extend(runner.run(steps)), daemon=True)
        t.start()
        while t.is_alive():  # X-Plane's flight loop
            plugin._flight_loop()
            t.join(0.001)
    finally:
        plugin.XPluginStop()
    return results, xp


def _transponder(plugin, xp):
    """The Baron's transponder: X-Plane's own mode commands."""
    mode = "sim/cockpit2/radios/actuators/transponder_mode"
    for name, n in (("off", 0), ("standby", 1), ("on", 2), ("alt", 3)):
        xp.effects[f"sim/transponder/transponder_{name}"] = lambda x, n=n: x.datarefs.update(
            {mode: n}
        )


def test_the_runway_checklists_pass_on_the_fake_baron(monkeypatch, tmp_path):
    steps = [s for s in checklists.ground("baron") if not s.name.startswith(("parked", "shutdown"))]
    results, xp = _run(monkeypatch, tmp_path, steps, setup=_transponder)
    failed = {r["step"]: r["problems"] for r in results if not r["ok"]}
    assert failed == {}
    by = {r["step"]: r["checklist"] for r in results}
    first = by["before takeoff checklist"]
    assert first["summary"] == "Before takeoff checklist complete"
    assert {a["answer"] for a in first["answers"]} == {"confirm"}
    assert by["before takeoff again"]["answers"] == []  # everything set reads right
    declined = by["approach, one declined, one not answered"]
    assert [(a["item"], a["answer"]) for a in declined["answers"]] == [("landing lights", "cancel")]
    assert declined["end"]["open"] == ["landing lights"]
    held = by["after takeoff on the ground, the gear held"]
    assert held["answers"][0]["hold"]["outcome"] == "cancelled"
    assert held["end"]["open"] == ["gear"]
    assert "sim/flight_controls/landing_gear_up" not in xp.commands_fired
    assert by["before landing cancelled"]["end"]["reason"] == "cancelled"
    line = scenario.line(results[0])
    assert line.startswith("✓ before takeoff checklist: Before takeoff checklist complete (")


def test_an_offer_that_doesnt_set_is_caught(monkeypatch, tmp_path):
    # The transponder's command does nothing here: the offer is "set" (it
    # went ahead), but asked again the item is offered once more.
    steps = [s for s in checklists.ground("baron") if s.name.startswith("before takeoff")]
    results, _ = _run(monkeypatch, tmp_path, steps)
    again = results[1]
    assert not again["ok"]
    assert "transponder: set last time, now offered" in again["problems"]


def test_a_hold_that_wasnt_wanted_is_a_failure(monkeypatch, tmp_path):
    steps = [
        checklists._start("after takeoff", scenario.Follow("after takeoff"), "after takeoff"),
    ]
    results, xp = _run(monkeypatch, tmp_path, steps)
    r = results[0]
    assert not r["ok"] and r["problems"][0].startswith("gear: held by a safety rule")
    assert r["checklist"]["answers"][0]["hold"]["outcome"] == "cancelled"
    assert "sim/flight_controls/landing_gear_up" not in xp.commands_fired


def test_the_report_counts_the_spoken_answers():
    results = [
        {"step": "x", "heard_ok": True, "understood_ok": True},
        {
            "step": "y",
            "checklist": {
                "answers": [
                    {"item": "gear", "heard_ok": False, "understood_ok": True},
                    {"item": "flaps", "answer": "wait"},
                ]
            },
        },
    ]
    total = scenario.summary([{**r, "ok": True} for r in results])
    assert (total["said_by_voice"], total["heard_exactly"], total["understood_by_voice"]) == (
        2,
        1,
        2,
    )
    assert json.dumps(total)


# ---- your own voice: simtest record checklists-ground / checklists-air ----
def test_record_lists_the_checklists_and_their_answers():
    from simtest import record

    words = [p for p, _ in record.phrases("checklists-ground", list(tier1.FLEET))]
    assert words == [
        "before takeoff checklist",
        "confirm",
        "cancel",
        "after landing checklist",
        "approach checklist",
        "after takeoff checklist",
        "before landing checklist",
        "shutdown checklist",
    ]
    air = [p for p, _ in record.phrases("checklists-air", ["737"])]
    assert air[:2] == ["autopilot on", "auto throttle on"]  # hands off first
    assert {"confirm", "cancel", "approach checklist"} <= set(air)
    assert len(air) == len(set(air))


def test_recorded_answers_are_played_for_the_offers(tmp_path):
    # The runner's own say: a recorded "confirm" is what goes over the bridge.
    sent = []

    class Bridge:
        def say_wav(self, wav, **_):
            sent.append(Path(wav).name)
            return {"ok": True, "heard": "confirm", "outcome": "executed", "replies": ["x"]}

        def say_text(self, text, **_):
            sent.append(text)
            return {"ok": True, "outcome": "executed", "replies": ["x"]}

    runner = scenario.Runner(Bridge(), voice=None, recordings={"confirm": tmp_path / "002.wav"})
    problems: list = []
    a = runner._reply(scenario.Follow("approach"), "landing lights", "confirm", problems)
    assert sent == ["002.wav"] and a["heard_ok"] and problems == []
    runner._reply(scenario.Follow("approach"), "landing lights", "cancel", problems)
    assert sent[-1] == "cancel" and runner.results == []  # not recorded: sent as text
