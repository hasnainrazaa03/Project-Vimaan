"""Tests for vimaan_nlu.overlay_state (what the in-sim overlay shows)."""

import pytest
from vimaan_nlu.overlay_state import HOLD_SEC, OverlayState, describe_slots


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def run(state, *events):
    for ev in events:
        state.on_event(ev)


def command(state, cmd="c0001", text="heading one eighty", outcome="executed", reply=None):
    run(
        state,
        {"kind": "ptt_press", "cmd": cmd},
        {"kind": "ptt_release", "cmd": cmd},
        {"kind": "stage", "stage": "stt", "cmd": cmd, "text": text, "backend": "vosk", "ms": 2.6},
        {
            "kind": "stage",
            "stage": "nlu",
            "cmd": cmd,
            "intent": "set_autopilot_heading",
            "slots": {"degrees": "180"},
            "confidence": 0.998,
        },
        {"kind": "speak", "cmd": cmd, "text": reply or "Heading one eight zero"},
        {"kind": "command_done", "cmd": cmd, "outcome": outcome, "speech_to_action_ms": 331.0},
    )


def test_hidden_until_used():
    s = OverlayState(Clock())
    assert not s.visible


def test_listening_shows_live_partial():
    s = OverlayState(Clock())
    s.on_event({"kind": "ptt_press", "cmd": "c1"})
    assert s.visible
    assert [(ln.badge, ln.text, ln.tone) for ln in s.lines()] == [("REC", "Listening...", "live")]
    s.on_partial("c1", "flight level two")
    assert s.lines()[0].text == "flight level two"
    s.on_partial("c9", "someone else")  # a stale command's partial is ignored
    assert s.lines()[0].text == "flight level two"


def test_processing_keeps_last_partial():
    s = OverlayState(Clock())
    run(s, {"kind": "ptt_press", "cmd": "c1"})
    s.on_partial("c1", "gear up")
    run(s, {"kind": "ptt_release", "cmd": "c1"})
    assert s.lines()[0].badge == "..." and s.lines()[0].text == "gear up"


def test_full_command():
    s = OverlayState(Clock())
    command(s)
    heard, meant, outcome = s.lines()
    assert heard.badge == "HEARD" and heard.text == '"heading one eighty"'
    assert heard.detail == "vosk  3 ms"
    assert meant.badge == "MEANT" and meant.text == "Heading   hdg 180" and meant.detail == "100%"
    assert outcome.badge == "DONE" and outcome.tone == "ok"
    assert outcome.text == "Heading one eight zero" and outcome.detail == "331 ms"


def test_fades_after_hold():
    clock = Clock()
    s = OverlayState(clock)
    command(s)
    clock.now += HOLD_SEC - 0.1
    assert s.visible
    clock.now += 0.2
    assert not s.visible


def test_verification_extends_and_marks():
    clock = Clock()
    s = OverlayState(clock)
    command(s)
    clock.now += HOLD_SEC - 1
    s.on_event({"kind": "verify", "cmd": "c0001", "label": "x", "ok": True})
    clock.now += 2
    assert s.visible and s.lines()[-1].detail.startswith("verified")


def test_failed_readback_is_red():
    s = OverlayState(Clock())
    command(s, reply="Autopilot 1 on")
    s.on_event({"kind": "verify", "cmd": "c0001", "label": "Autopilot 1", "ok": False})
    s.on_event({"kind": "speak", "cmd": "c0001", "text": "Autopilot 1 did not respond"})
    last = s.lines()[-1]
    assert (last.badge, last.tone, last.text) == (
        "NO RESPONSE",
        "error",
        "Autopilot 1 did not respond",
    )


@pytest.mark.parametrize(
    ("outcome", "badge", "tone"),
    [
        ("safety_hold", "CONFIRM?", "warn"),
        ("ambiguous", "WHICH?", "warn"),
        ("rejected", "REJECTED", "warn"),
        ("not_understood", "NOT HEARD", "error"),
        ("timeout", "TIMEOUT", "error"),
        ("something_new", "ERROR", "error"),
    ],
)
def test_outcome_badges(outcome, badge, tone):
    s = OverlayState(Clock())
    command(s, outcome=outcome)
    assert (s.lines()[-1].badge, s.lines()[-1].tone) == (badge, tone)


def test_rejection_reason_shown():
    s = OverlayState(Clock())
    run(s, {"kind": "ptt_press", "cmd": "c1"}, {"kind": "ptt_release", "cmd": "c1"})
    run(
        s,
        {"kind": "stage", "stage": "stt", "cmd": "c1", "text": "seventy", "backend": "vosk"},
        {"kind": "implausible", "cmd": "c1", "reason": "no_command_word"},
        {"kind": "speak", "cmd": "c1", "text": "Please repeat that"},
        {"kind": "command_done", "cmd": "c1", "outcome": "rejected"},
    )
    assert s.lines()[-1].text == "Please repeat that  (no command word heard)"


def test_low_confidence_meaning_is_amber():
    s = OverlayState(Clock())
    run(s, {"kind": "ptt_press", "cmd": "c1"})
    run(
        s,
        {
            "kind": "stage",
            "stage": "nlu",
            "cmd": "c1",
            "intent": "set_flaps",
            "slots": {},
            "confidence": 0.4,
        },
    )
    run(s, {"kind": "command_done", "cmd": "c1", "outcome": "low_confidence"})
    assert s.lines()[1].tone == "warn"


def test_error_before_transcript_still_shows_what_happened():
    s = OverlayState(Clock())
    run(
        s,
        {"kind": "ptt_press", "cmd": "c1"},
        {"kind": "speak", "cmd": None, "text": "I could not understand you"},
        {"kind": "command_done", "cmd": "c1", "outcome": "not_understood"},
    )
    lines = s.lines()
    assert lines[0].text == "Recognising..."
    assert (lines[-1].badge, lines[-1].text) == ("NOT HEARD", "I could not understand you")


def test_events_for_other_commands_ignored():
    s = OverlayState(Clock())
    command(s, cmd="c2")
    s.on_event({"kind": "stage", "stage": "stt", "cmd": "c1", "text": "old", "backend": "vosk"})
    assert s.heard == "heading one eighty"


def test_spoken_notice_when_idle():
    clock = Clock()
    s = OverlayState(clock)
    s.on_event({"kind": "speak", "cmd": None, "text": "Engine 1 did not start"})
    assert s.visible and s.lines()[0].text == "Engine 1 did not start"
    clock.now += HOLD_SEC + 1
    assert not s.visible


def test_modes():
    clock = Clock()
    s = OverlayState(clock)
    assert s.cycle_mode() == "pinned" and s.visible  # pinned shows even when idle
    assert s.lines()[0].badge == "VIMAAN"
    assert s.cycle_mode() == "off"
    command(s)
    assert not s.visible
    assert s.cycle_mode() == "auto"
    assert OverlayState(clock, mode="bogus").mode == "auto"


def test_notice_does_not_interrupt_a_recording():
    s = OverlayState(Clock())
    s.on_event({"kind": "ptt_press", "cmd": "c1"})
    s.notice("Overlay pinned")
    assert s.phase == "listening"


def test_describe_slots():
    assert describe_slots({"com_port": "1", "frequency": "118.75"}) == "COM 1  freq 118.75"
    assert describe_slots({"state": "on"}) == "ON"


def test_answered_reply_shows_what_it_was_taken_as():
    s = OverlayState(Clock())
    run(
        s,
        {"kind": "ptt_press", "cmd": "c2"},
        {"kind": "ptt_release", "cmd": "c2"},
        {"kind": "stage", "stage": "stt", "cmd": "c2", "text": "one thousand", "backend": "vosk"},
        {"kind": "stage", "stage": "nlu", "cmd": "c2", "intent": "None", "confidence": 1.0},
        {
            "kind": "answered",
            "cmd": "c2",
            "reply_to": "c1",
            "intent": "set_autopilot_altitude",
            "slots": {"altitude": "1000"},
        },
        {"kind": "speak", "cmd": "c2", "text": "Altitude 1000 feet"},
        {"kind": "command_done", "cmd": "c2", "outcome": "executed"},
    )
    _, meant, outcome = s.lines()
    assert meant.text.endswith("alt 1000") and meant.detail == ""
    assert outcome.badge == "DONE"


def test_bare_number_reason_shown():
    s = OverlayState(Clock())
    run(
        s,
        {"kind": "ptt_press", "cmd": "c1"},
        {"kind": "implausible", "cmd": "c1", "reason": "not_offered"},
        {"kind": "speak", "cmd": "c1", "text": "Say it with the word heading"},
        {"kind": "command_done", "cmd": "c1", "outcome": "rejected"},
    )
    assert s.lines()[-1].text == "Say it with the word heading  (not one of the two values offered)"


# ---------------------------------------------------------------------------
# The aircraft probe's own view
# ---------------------------------------------------------------------------
def start_probe(s, steps=15):
    s.on_event(
        {
            "kind": "probe_start",
            "where": "air",
            "steps": steps,
            "name": "BE58",
            "next": "heading mode",
        }
    )


def test_probe_view_shows_progress_and_whats_next():
    s = OverlayState(Clock())
    start_probe(s)
    title, ready = s.lines()
    assert (title.badge, title.text, title.detail, title.progress) == (
        "PROBE",
        "BE58  air probe",
        "0 / 15",
        0.0,
    )
    assert (ready.badge, ready.text) == ("NEXT", "heading mode  (press the probe key)")
    s.on_event({"kind": "probe_act", "label": "heading mode", "index": 0, "total": 15})
    ln = s.lines()[1]
    assert (ln.badge, ln.text, ln.tone) == ("TESTING", "heading mode", "live")
    s.on_event(
        {
            "kind": "probe_step",
            "label": "heading mode",
            "verdict": "works",
            "index": 0,
            "next": "nav mode",
        }
    )
    title, result, nxt = s.lines()
    assert title.detail == "1 / 15" and title.progress == pytest.approx(1 / 15)
    assert (result.badge, result.text, result.tone) == ("WORKS", "heading mode", "ok")
    assert nxt.text == "nav mode  (press the probe key)"


def test_probe_results_never_mix_with_the_last_command():
    # In flight the probe's announcements replaced the last command's reply
    # line, under its HEARD / MEANT lines.
    s = OverlayState(Clock())
    command(s)  # "heading one eighty" ... DONE
    start_probe(s)
    s.on_event(
        {"kind": "speak", "cmd": None, "text": "Probe ready. 15 air steps.", "origin": "probe"}
    )
    lines = s.lines()
    assert [ln.badge for ln in lines] == ["PROBE", "NEXT"]
    assert all("heading one eighty" not in ln.text for ln in lines)
    assert s.reply == "Heading one eight zero"  # the command's own record is untouched


def test_a_voice_command_takes_over_then_hands_back_to_the_probe():
    clock = Clock()
    s = OverlayState(clock)
    start_probe(s)
    command(s, text="autopilot on")
    assert s.lines()[0].badge == "HEARD"
    clock.now += HOLD_SEC + 0.1
    assert s.visible and s.lines()[0].badge == "PROBE"


def test_probe_view_stays_up_between_presses():
    clock = Clock()
    s = OverlayState(clock)
    start_probe(s)
    clock.now += 600  # the pilot takes their time between steps
    assert s.visible


def test_finished_probe_shows_a_summary_then_hides():
    clock = Clock()
    s = OverlayState(clock)
    start_probe(s, steps=2)
    for i, verdict in enumerate(("works", "no_change")):
        s.on_event({"kind": "probe_step", "label": f"step {i}", "verdict": verdict, "index": i})
    s.on_event({"kind": "probe_end", "reason": "complete", "message": "Probe complete."})
    ln = s.lines()[-1]
    assert (ln.badge, ln.text, ln.tone) == ("DONE", "Results saved. 1 work, 1 no change.", "ok")
    clock.now += 13
    assert not s.visible


def test_stopped_probe_is_red():
    s = OverlayState(Clock())
    start_probe(s)
    s.on_event(
        {
            "kind": "probe_end",
            "reason": "restore_failed",
            "message": "Probe stopped. Heading mode did not restore. Check the autopilot.",
        }
    )
    ln = s.lines()[-1]
    assert (ln.badge, ln.tone) == ("STOPPED", "error")
    assert ln.text == "Probe stopped. Heading mode did not restore. Check the autopilot."
