"""The automated-test bridge's file protocol (vimaan_nlu.sim_bridge)."""

import json
import os

import pytest
from vimaan_nlu import sim_bridge as b


def test_paths(tmp_path):
    flag, inbox, outbox = b.paths(tmp_path)
    assert flag == tmp_path / "test" / "bridge.on"
    assert (inbox.name, outbox.name) == ("inbox", "outbox")


def test_requests_are_taken_oldest_first_and_removed(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for i, name in enumerate(("b", "a")):
        p = inbox / f"{name}.json"
        p.write_text(json.dumps({"id": name, "op": "state"}))
        os.utime(p, ns=(i * 10**9, i * 10**9))
    (inbox / "half.json.tmp").write_text("{")  # still being written
    (inbox / "bad.json").write_text("{not json")
    os.utime(inbox / "bad.json", ns=(5 * 10**9, 5 * 10**9))
    got = b.take_requests(inbox)
    assert [r["id"] for r in got] == ["b", "a", "bad"]
    assert "unreadable" in got[2]["error"]
    assert sorted(p.name for p in inbox.iterdir()) == ["half.json.tmp"]


def test_results_are_written_atomically(tmp_path):
    b.write_json(tmp_path / "out" / "r1.json", {"id": "r1", "ok": True})
    assert json.loads((tmp_path / "out" / "r1.json").read_text())["ok"] is True
    assert not list((tmp_path / "out").glob("*.tmp"))


@pytest.mark.parametrize(
    ("req", "error"),
    [
        ({"id": "x", "op": "state"}, None),
        ({"id": "../x", "op": "state"}, "bad id"),
        ({"id": "x", "op": "fly"}, "unknown op"),
        ({"id": "x", "op": "say"}, "exactly one"),
        ({"id": "x", "op": "say", "text": "a", "wav": "/a.wav"}, "exactly one"),
        ({"id": "x", "op": "say", "wav": "rel.wav"}, "absolute"),
        ({"id": "x", "op": "say", "wav": "/nope/missing.wav"}, "no such file"),
        ({"id": "x", "op": "say", "text": "gear up"}, None),
        ({"id": "x", "op": "read", "datarefs": []}, "list"),
        ({"id": "x", "op": "write", "dataref": "a", "value": "1"}, "number"),
        ({"id": "x", "op": "command", "name": "a", "phase": "hold"}, "phase"),
        ({"id": "x", "op": "place", "latitude": 1, "longitude": 2}, "elevation_ft"),
        ({"id": "x", "op": "airport", "icao": "ks fo"}, "ICAO"),
        ({"id": "x", "op": "aircraft", "path": "/a/b.obj"}, ".acf"),
        ({"id": "x", "error": "unreadable request: x"}, "unreadable"),
    ],
)
def test_problems(req, error):
    got = b.problem(req)
    assert got is None if error is None else error in got


def test_parse_dataref():
    assert b.parse_dataref("sim/a/b[3]") == ("sim/a/b", 3)
    assert b.parse_dataref("sim/a/b") == ("sim/a/b", None)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_a_say_result_waits_for_its_read_backs():
    clock = Clock()
    r = b.SayResults(clock)
    r.start("req1", "c0001", now=0.0)
    for ev in (
        {
            "kind": "stage",
            "stage": "nlu",
            "cmd": "c0001",
            "intent": "toggle_autopilot",
            "slots": {"state": "on"},
            "confidence": 0.99,
        },
        {
            "kind": "sim_command",
            "cmd": "c0001",
            "command": "sim/autopilot/servos_toggle",
            "ok": True,
        },
        {"kind": "speak", "cmd": "c0001", "text": "Autopilot 1 on"},
        {
            "kind": "command_done",
            "cmd": "c0001",
            "outcome": "executed",
            "speech_to_action_ms": 3.0,
            "text": "autopilot on",
        },
        {"kind": "heartbeat"},
        {"kind": "speak", "cmd": "c0002", "text": "someone else's"},
    ):
        r.on_event(ev)
    assert r.busy and r.finished(0.5, checking={"c0001"}) == []  # a read-back is due
    r.on_event(
        {
            "kind": "verify",
            "cmd": "c0001",
            "label": "Autopilot 1",
            "expected": 1,
            "actual": 1,
            "ok": True,
        }
    )
    (res,) = r.finished(1.0, checking=set())
    assert res["id"] == "req1" and res["ok"] and res["outcome"] == "executed"
    assert res["heard"] == "autopilot on" and res["replies"] == ["Autopilot 1 on"]
    assert res["actions"] == [{"command": "sim/autopilot/servos_toggle", "ok": True}]
    assert res["checks"][0]["ok"] is True and not r.busy


def test_read_backs_and_answers_dont_wait_forever():
    clock = Clock()
    r = b.SayResults(clock)
    r.start("a", "c1", now=0.0)
    r.on_event({"kind": "command_done", "cmd": "c1", "outcome": "executed"})
    assert r.finished(b.CHECKS_WAIT_SEC - 1, checking={"c1"}) == []
    assert r.finished(b.CHECKS_WAIT_SEC + 1, checking={"c1"})[0]["outcome"] == "executed"
    r.start("b", "c2", now=0.0)
    (late,) = r.finished(b.SAY_TIMEOUT_SEC + 1, checking=set())
    assert late["ok"] is False and late["error"] == "timeout"
