"""Tests for vimaan_nlu.telemetry (plugin diagnostics)."""

import json
import sys
import threading

import pytest
from vimaan_nlu import telemetry
from vimaan_nlu.telemetry import Telemetry


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def tel(tmp_path):
    clock = FakeClock()
    t = Telemetry(tmp_path, clock=clock)
    t.fake_clock = clock
    yield t
    if t._faults is not None:
        t.close()


def read_events(t):
    with open(t.events_path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_creates_session_files(tel):
    assert tel.events_path.exists()
    assert tel.log_path.exists()
    assert tel.faults_path.exists()


def test_event_is_flushed_immediately(tel):
    tel.event("hello", value=1)
    # Readable before close(): a crash right after must not lose it.
    assert read_events(tel)[-1]["kind"] == "hello"


def test_stage_records_duration(tel):
    with tel.stage("nlu", cmd="c0001") as res:
        tel.fake_clock.now += 0.012
        res["intent"] = "gear"
    ev = read_events(tel)[-1]
    assert ev["stage"] == "nlu" and ev["cmd"] == "c0001" and ev["intent"] == "gear"
    assert ev["ms"] == pytest.approx(12.0)
    assert "slow" not in ev


def test_stage_over_budget_is_flagged_slow(tel):
    with tel.stage("nlu"):
        tel.fake_clock.now += 1.0
    ev = read_events(tel)[-1]
    assert ev["slow"] is True and ev["budget_ms"] == telemetry.STAGE_BUDGET_MS["nlu"]
    assert tel.counters["slow"] == 1


def test_stage_failure_logs_traceback_and_reraises(tel):
    with pytest.raises(ValueError), tel.stage("dispatch", cmd="c0002"):
        raise ValueError("boom")
    ev = read_events(tel)[-1]
    assert ev["kind"] == "stage_failed" and ev["error_type"] == "ValueError"
    assert "boom" in ev["traceback"]
    assert tel.counters["errors"] == 1


def test_expected_exception_is_not_an_error(tel):
    class Timeout(Exception):
        pass

    with pytest.raises(Timeout), tel.stage("capture", expected=(Timeout,)):
        raise Timeout()
    ev = read_events(tel)[-1]
    assert ev["kind"] == "stage_ended" and ev["reason"] == "Timeout"
    assert tel.counters["errors"] == 0


def test_command_done_flags_slow_total(tel):
    start = tel.fake_clock.now
    tel.fake_clock.now += 5.0
    tel.command_done("c0003", "executed", start)
    ev = read_events(tel)[-1]
    assert ev["total_ms"] == pytest.approx(5000.0) and ev["slow"] is True


def test_tick_detects_sim_stall_and_overrun(tel):
    tel.tick(busy_ms=1.0)
    tel.fake_clock.now += telemetry.STALL_GAP_SEC + 1
    tel.tick(busy_ms=telemetry.MAIN_THREAD_BUDGET_MS + 5)
    kinds = [e["kind"] for e in read_events(tel)]
    assert "sim_stall" in kinds and "main_thread_overrun" in kinds
    assert tel.counters["stalls"] == 1


def test_heartbeat_emitted_periodically(tel):
    tel.fake_clock.now += telemetry.HEARTBEAT_SEC + 1
    tel.tick(queue_depth=3)
    hb = [e for e in read_events(tel) if e["kind"] == "heartbeat"]
    assert hb and hb[-1]["queue_depth"] == 3


# The crash is deliberate; pytest also sees it through the hook we chain to.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_uncaught_thread_exception_recorded(tel):
    def worker():
        raise RuntimeError("worker died")

    t = threading.Thread(target=worker, name="listen-c0004")
    t.start()
    t.join()
    ev = [e for e in read_events(tel) if e["kind"] == "uncaught_exception"]
    assert ev and ev[-1]["thread"] == "listen-c0004" and "worker died" in ev[-1]["error"]


def test_close_restores_global_hooks(tmp_path):
    before_sys, before_thread = sys.excepthook, threading.excepthook
    t = Telemetry(tmp_path)
    assert sys.excepthook is not before_sys
    t.close()
    assert sys.excepthook is before_sys and threading.excepthook is before_thread
    assert read_events(t)[-1]["kind"] == "session_end"


def test_mirror_receives_log_lines(tmp_path):
    lines = []
    t = Telemetry(tmp_path, mirror=lines.append, install_crash_handlers=False)
    t.event("ping")
    t.close()
    assert any("[ping]" in line for line in lines)


def test_command_ids_are_sequential(tel):
    assert tel.new_command_id() == "c0001"
    assert tel.new_command_id() == "c0002"


def test_command_done_ignores_key_hold_time(tel):
    start = tel.fake_clock.now
    tel.fake_clock.now += 6.0  # pilot held the key for ~5 s
    tel.command_done("c0005", "executed", start, speech_to_action_ms=900.0)
    assert "slow" not in read_events(tel)[-1]
