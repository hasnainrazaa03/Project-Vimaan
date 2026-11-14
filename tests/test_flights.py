"""The dashboard's Flights view (dashboard/flights.py and its API), on
synthetic session logs shaped like the plugin's telemetry."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard import flights, paths  # noqa: E402

SESSION = "20261111_101500"


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FLIGHT_LOGS", tmp_path)
    flights._cache.clear()
    (tmp_path / "audio").mkdir()
    return tmp_path


def ev(t, kind, **fields):
    return {"ts": f"2026-11-11T10:15:{int(t) % 60:02d}.000", "t": t, "kind": kind, **fields}


def command(t, cmd, heard, intent, slots, outcome, reply, **extra):
    return [
        ev(t, "ptt_press", cmd=cmd, source="hotkey"),
        ev(t + 1, "ptt_release", cmd=cmd, held_ms=1200.0),
        ev(t + 1.3, "stage", stage="capture", cmd=cmd, audio_sec=1.2, audio_file=f"/x/{cmd}.wav"),
        ev(t + 1.3, "stage", stage="stt", cmd=cmd, ms=3.0, text=heard),
        ev(
            t + 1.3,
            "stage",
            stage="nlu",
            cmd=cmd,
            ms=6.0,
            normalized=heard,
            intent=intent,
            confidence=0.99,
            slots=slots,
        ),
        *extra.get("middle", []),
        ev(t + 1.3, "speak", cmd=cmd, text=reply),
        ev(t + 1.3, "stage", stage="dispatch", cmd=cmd, intent=intent, outcome=outcome),
        ev(
            t + 1.3,
            "command_done",
            cmd=cmd,
            outcome=outcome,
            total_ms=1500.0,
            speech_to_action_ms=extra.get("latency", 320.0),
            text=heard,
        ),
    ]


def session_rows(end=True):
    rows = [
        ev(0.1, "session_start", backend="torch", stt_backend="vosk"),
        ev(2.0, "stage", stage="model_load", ms=3570.0, model_path="/m/vimaan_nlu_model_best/v19"),
        ev(
            3.0,
            "aircraft",
            icao="BE58",
            acf="/X-Plane 12/Aircraft/Laminar Research/Beechcraft Baron 58/Baron_58.acf",
        ),
        ev(3.0, "joystick", has_joystick=1, nullzone=0.0, stick={"pitch": 0.0}),
        ev(4.0, "stage", stage="self_test", passed=15, total=15),
        *command(
            10,
            "c0001",
            "autopilot on",
            "toggle_autopilot",
            {"state": "on"},
            "executed",
            "Autopilot 1 on",
            middle=[
                ev(
                    11.3, "sim_command", cmd="c0001", command="sim/autopilot/servos_toggle", ok=True
                ),
                ev(12.3, "verify", cmd="c0001", label="Autopilot 1", expected=1, actual=1, ok=True),
            ],
        ),
        ev(
            11.5,
            "autopilot_state_change",
            changed={"servos_on": [0, 1]},
            last_vimaan_cmd="c0001",
            last_vimaan_intent="toggle_autopilot",
            sec_since_vimaan_action=0.2,
        ),
        *command(
            20, "c0002", "yet ap", "toggle_autopilot", {}, "rejected", "On or off?", latency=400.0
        ),
        *command(30, "c0003", "", None, {}, "too_short", "", latency=None),
        # 31 s after Vimaan's last action, all off: not Vimaan.
        ev(
            52.0,
            "autopilot_state_change",
            changed={"servos_on": [1, 0], "flight_director": [2, 0], "vertical_mode": [6, 3]},
            last_vimaan_cmd="c0002",
            last_vimaan_intent="toggle_autopilot",
            sec_since_vimaan_action=30.7,
        ),
        ev(60.0, "probe_start", where="air", steps=15),
        ev(
            61.0,
            "autopilot_state_change",
            changed={"heading_status": [2, 0]},
            last_vimaan_cmd="c0002",
            last_vimaan_intent="toggle_autopilot",
            sec_since_vimaan_action=40.0,
        ),
        ev(62.0, "probe_step", role="mode_heading", verdict="works"),
        ev(63.0, "probe_end", reason="complete", steps={"mode_heading": "works"}),
        ev(70.0, "sim_stall", gap_ms=3818),
    ]
    if end:
        rows.append(ev(80.0, "session_end"))
    return rows


def write(logs, rows, session=SESSION, torn=""):
    p = logs / f"vimaan_{session}.events.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) + torn)
    return p


def test_a_session_becomes_commands_and_events(logs):
    write(logs, session_rows(), torn='{"ts": "2026-11-11T10:16')  # a torn last line
    d = flights.load(SESSION)
    s = d["summary"]
    assert s["status"] == "ended" and s["model"] == "v19" and s["stt"] == "vosk"
    assert s["aircraft"] == ["Beechcraft Baron 58"]
    assert s["commands"] == 3 and s["recordings"] == 3
    assert s["groups"]["done"] == 1 and s["groups"]["refused"] == 1 and s["groups"]["missed"] == 1
    assert s["latency_p50_ms"] == 360.0
    assert s["autopilot_drops"] == 1  # the unexplained one, not Vimaan's nor the probe's
    assert s["problems"] == 1 and s["self_test"] == {"passed": 15, "total": 15}

    cmds = [i for i in d["items"] if i["type"] == "command"]
    c = cmds[0]
    assert (c["heard"], c["intent_name"], c["label"], c["replies"]) == (
        "autopilot on",
        "Autopilot",
        "Done",
        ["Autopilot 1 on"],
    )
    assert c["actions"] == [{"kind": "command", "name": "sim/autopilot/servos_toggle", "ok": True}]
    assert c["checks"][0]["ok"] is True and c["held_ms"] == 1200.0 and c["audio"]
    assert cmds[1]["label"] == "Not done" and cmds[1]["replies"] == ["On or off?"]

    changes = [i for i in d["items"] if i.get("kind") == "autopilot_state_change"]
    assert [(e["who"].split(" (")[0], e["tone"], e["level"]) for e in changes] == [
        ("Vimaan", "info", "autopilot"),
        ("not Vimaan", "warn", "autopilot"),
        ("the probe", "info", "system"),  # probe presses are details
    ]
    assert changes[1]["changes"][0] == {"name": "Autopilot", "from": "on", "to": "off"}
    assert {"name": "Vertical mode", "from": "altitude hold", "to": "pitch"} in changes[1][
        "changes"
    ]
    texts = [i["text"] for i in d["items"] if i["type"] == "event"]
    assert any(t.startswith("Joystick null zone is 0%") for t in texts)
    assert "X-Plane paused or froze for 3.8 s" in texts


def test_the_plugins_own_disconnect_diagnosis_is_used_when_present(logs):
    rows = session_rows()
    rows.insert(
        -1,
        ev(
            52.1,
            "autopilot_disconnect",
            cause="stick",
            detail="stick pitch +0.62",
            said="Autopilot disconnected. Stick moved.",
            stick_peak={"pitch": 0.62},
            nullzone=0.0,
            commands=[],
        ),
    )
    write(logs, rows)
    d = flights.load(SESSION)
    assert d["summary"]["autopilot_drops"] == 1  # not counted twice
    (e,) = [i for i in d["items"] if i.get("kind") == "autopilot_disconnect"]
    assert e["tone"] == "bad" and e["cause"] == "stick" and "+0.62" in e["text"]


def test_live_and_unfinished_sessions(logs):
    p = write(logs, session_rows(end=False))
    assert flights.load(SESSION)["summary"]["status"] == "live"
    old = time.time() - flights.LIVE_SEC - 10
    os.utime(p, (old, old))
    assert flights.load(SESSION)["summary"]["status"] == "crashed"


def test_flights_are_listed_newest_first(logs):
    write(logs, session_rows(), "20261101_090000")
    write(logs, session_rows(), "20261111_101500")
    (logs / "vimaan_notasession.events.jsonl").write_text("{}\n")
    assert [f["session"] for f in flights.list_flights()] == ["20261111_101500", "20261101_090000"]


@pytest.mark.parametrize("bad", ["../x", "", "2026111_1015", "20261111_101500/../../x", None])
def test_names_that_are_not_sessions(logs, bad):
    assert flights.events_path(bad) is None and flights.load(bad) is None


def test_recordings_are_found_only_by_session_and_command(logs):
    (logs / "audio" / f"{SESSION}_c0001.wav").write_bytes(b"RIFF")
    assert flights.audio_path(SESSION, "c0001").name == f"{SESSION}_c0001.wav"
    for session, cmd in (
        (SESSION, "c1"),
        (SESSION, "../c0001"),
        ("..", "c0001"),
        (SESSION, "c0002"),
    ):
        assert flights.audio_path(session, cmd) is None


# ---- API -------------------------------------------------------------------
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(logs):
    from dashboard.app import app

    return TestClient(app)


def test_api(client, logs):
    write(logs, session_rows())
    (logs / "audio" / f"{SESSION}_c0001.wav").write_bytes(b"RIFF....WAVE")
    listed = client.get("/api/flights").json()
    assert [f["session"] for f in listed["flights"]] == [SESSION]
    assert client.get(f"/api/flights/{SESSION}").json()["summary"]["commands"] == 3
    r = client.get(f"/api/flights/{SESSION}/stream?follow=false")
    assert r.headers["content-type"].startswith("text/event-stream")
    first = r.text.split("\n\n")[0]
    assert first.startswith("event: flight\ndata: ")
    assert json.loads(first.split("data: ", 1)[1])["session"] == SESSION
    a = client.get(f"/api/flights/{SESSION}/audio/c0001")
    assert a.status_code == 200 and a.headers["content-type"] == "audio/wav"


@pytest.mark.parametrize(
    "path",
    [
        "/api/flights/20990101_000000",
        "/api/flights/nope/stream",
        f"/api/flights/{SESSION}/audio/c0009",
        f"/api/flights/{SESSION}/audio/..%2F..%2Fx",
        "/api/flights/..%2F..%2Fetc",
    ],
)
def test_unknown_flights_and_recordings_are_404(client, logs, path):
    write(logs, session_rows())
    assert client.get(path).status_code == 404


def test_transponder_selector_steps_are_noted(logs):
    rows = [
        ev(0.1, "session_start", backend="torch", stt_backend="vosk"),
        *command(
            10,
            "c0001",
            "transponder alt",
            "set_transponder_mode",
            {"xpdr_mode": "alt"},
            "pending",
            "Transponder altitude",
            middle=[
                ev(11.3, "knob_step_start", cmd="c0001", mode="alt", knobs=1),
                ev(11.4, "knob_step_end", cmd="c0001", ok=True, presses=2, why=None),
            ],
        ),
        *command(
            20,
            "c0002",
            "transponder standby",
            "set_transponder_mode",
            {"xpdr_mode": "standby"},
            "pending",
            "Transponder not set: the selector did not move.",
            middle=[
                ev(21.4, "knob_step_end", cmd="c0002", ok=False, presses=1, why="it is stuck"),
            ],
        ),
        ev(30.0, "session_end"),
    ]
    write(logs, rows)
    cmds = [i for i in flights.load(SESSION)["items"] if i["type"] == "command"]
    notes = [[(n["text"], n["tone"]) for n in c["notes"]] for c in cmds]
    assert notes[0] == [
        ("Turning the transponder selector to alt", "info"),
        ("Selector set after 2 clicks", "info"),
    ]
    assert notes[1] == [("Selector not set: it is stuck", "warn")]
