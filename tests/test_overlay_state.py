"""Tests for vimaan_nlu.overlay_state (what the in-sim overlay shows)."""

import pytest
from vimaan_nlu.overlay_state import (
    CHECKLIST_HOLD_SEC,
    CHECKLIST_RESULT_SEC,
    HISTORY_FADE_SEC,
    HISTORY_LINES,
    HISTORY_SEC,
    HOLD_SEC,
    WORKING_MAX_SEC,
    OverlayState,
    action_text,
    age_text,
    describe_slots,
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def run(state, *events):
    for ev in events:
        state.on_event(ev)


def command(
    state,
    cmd="c0001",
    text="heading one eighty",
    outcome="executed",
    reply=None,
    intent="set_autopilot_heading",
    slots=None,
    confidence=0.998,
    source="hotkey",
):
    run(
        state,
        {"kind": "ptt_press", "cmd": cmd, "source": source},
        {"kind": "ptt_release", "cmd": cmd},
        {"kind": "stage", "stage": "stt", "cmd": cmd, "text": text, "backend": "vosk", "ms": 2.6},
        {
            "kind": "stage",
            "stage": "nlu",
            "cmd": cmd,
            "intent": intent,
            "slots": {"degrees": "180"} if slots is None else slots,
            "confidence": confidence,
        },
        {"kind": "speak", "cmd": cmd, "text": reply or "Heading one eight zero"},
        {"kind": "command_done", "cmd": cmd, "outcome": outcome, "speech_to_action_ms": 331.0},
    )


def rows(state):
    return {r.label: r for r in state.panel().rows if r.label}


def tags(state):
    return [(t.text, t.tone, t.filled) for t in state.panel().tags]


def test_hidden_until_used():
    s = OverlayState(Clock())
    assert not s.visible


def test_listening_shows_live_partial():
    s = OverlayState(Clock())
    s.on_event({"kind": "ptt_press", "cmd": "c1", "source": "hotkey"})
    assert s.visible
    assert tags(s) == [("LISTEN", "live", True), ("KEY", "muted", False)]
    r = rows(s)
    assert (r["HEARD"].text, r["HEARD"].tone) == ("_", "heard")
    assert r["ACTION"].text == "----"
    assert r["CONF"].kind == "conf" and r["CONF"].value is None and r["CONF"].detail == "--"
    s.on_partial("c1", "flight level two")
    assert rows(s)["HEARD"].text == "flight level two_"
    s.on_partial("c9", "someone else")  # a stale command's partial is ignored
    assert rows(s)["HEARD"].text == "flight level two_"


@pytest.mark.parametrize(
    ("source", "tag", "tone"), [("wake", "WAKE", "ok"), ("test", "TEST", "muted")]
)
def test_where_the_command_came_from(source, tag, tone):
    s = OverlayState(Clock())
    s.on_event({"kind": "ptt_press", "cmd": "c1", "source": source})
    assert tags(s)[1] == (tag, tone, False)


def test_processing_keeps_last_partial():
    s = OverlayState(Clock())
    run(s, {"kind": "ptt_press", "cmd": "c1"})
    s.on_partial("c1", "gear up")
    run(s, {"kind": "ptt_release", "cmd": "c1"})
    assert tags(s)[0][0] == "WAIT" and rows(s)["HEARD"].text == "gear up"


def test_full_command():
    s = OverlayState(Clock())
    command(s)
    p = s.panel()
    assert tags(s)[0] == ("DONE", "ok", True) and p.meta == "331 MS" and not p.alert
    r = rows(s)
    assert (r["HEARD"].text, r["HEARD"].tone) == ("heading one eighty", "heard")
    assert (r["ACTION"].text, r["ACTION"].tone) == ("HDG 180", "text")
    conf = r["CONF"]
    assert (conf.value, conf.detail, conf.tone) == (0.998, "100%", "ok")
    assert "REPLY" not in r  # the read-back says what ACTION says


def test_low_confidence_is_amber_below_the_floor():
    s = OverlayState(Clock(), floor=0.55)
    command(
        s,
        text="flaps fifteen",
        intent="set_flaps",
        slots={"flap_setting": "15"},
        confidence=0.48,
        outcome="low_confidence",
        reply="Please repeat that",
    )
    p = s.panel()
    assert tags(s)[0] == ("REPEAT", "warn", True) and p.alert == "warn" and p.floor == 0.55
    r = rows(s)
    assert (r["ACTION"].text, r["ACTION"].tone, r["ACTION"].detail) == (
        "FLAPS 15?",
        "muted",
        "NOT DONE",
    )
    assert (r["CONF"].value, r["CONF"].tone, r["CONF"].detail) == (0.48, "warn", "48%")
    assert r["REPLY"].text == "PLEASE REPEAT THAT"


def test_result_holds_then_the_history_stays_then_hides():
    clock = Clock()
    s = OverlayState(clock)
    command(s)
    clock.now += HOLD_SEC - 0.1
    assert tags(s)[0][0] == "DONE"
    clock.now += 0.2
    assert s.visible  # the idle view, with the command in the history
    p = s.panel()
    assert [(h.tag, h.text, h.tone) for h in p.history] == [("DONE", "HDG 180", "ok")]
    assert p.history[0].age == "-0:06" and p.history[0].alpha == 1.0
    clock.now = 100.0 + HISTORY_SEC - HISTORY_FADE_SEC / 2
    assert s.panel().history[0].alpha == pytest.approx(0.5)
    clock.now = 100.0 + HISTORY_SEC + 0.1
    assert not s.visible


def test_history_shows_the_newest_three_and_not_the_one_on_screen():
    clock = Clock()
    s = OverlayState(clock)
    for i, deg in enumerate(("090", "180", "270", "360")):
        command(s, cmd=f"c{i}", slots={"degrees": deg})
        clock.now += 10
    command(s, cmd="c9", slots={"degrees": "045"})
    history = s.panel().history
    assert len(history) == HISTORY_LINES
    assert [h.text for h in history] == ["HDG 360", "HDG 270", "HDG 180"]
    assert [h.age for h in history] == ["-0:10", "-0:20", "-0:30"]


def test_age_text():
    assert (age_text(12.4), age_text(65), age_text(-1)) == ("-0:12", "-1:05", "-0:00")


def test_failed_readback_is_red():
    s = OverlayState(Clock())
    command(s, reply="Autopilot 1 on", intent="toggle_autopilot", slots={"state": "on"})
    s.on_event({"kind": "verify", "cmd": "c0001", "label": "Autopilot 1", "ok": False})
    s.on_event({"kind": "speak", "cmd": "c0001", "text": "Autopilot 1 did not respond"})
    p = s.panel()
    assert tags(s)[0] == ("NO RESPONSE", "error", True) and p.alert == "error"
    assert (rows(s)["REPLY"].text, rows(s)["REPLY"].tone) == (
        "AUTOPILOT 1 DID NOT RESPOND",
        "error",
    )
    assert (s.history[-1]["tag"], s.history[-1]["tone"]) == ("FAIL", "error")


def test_verification_extends_the_hold():
    clock = Clock()
    s = OverlayState(clock)
    command(s)
    clock.now += HOLD_SEC - 1
    s.on_event({"kind": "verify", "cmd": "c0001", "label": "x", "ok": True})
    clock.now += 2
    assert tags(s)[0][0] == "DONE"


@pytest.mark.parametrize(
    ("outcome", "status", "tone"),
    [
        ("safety_hold", "CAUTION", "warn"),
        ("ambiguous", "WHICH?", "warn"),
        ("rejected", "REFUSED", "warn"),
        ("no_op", "NO CHANGE", "ok"),
        ("not_understood", "NOT HEARD", "error"),
        ("timeout", "TIMEOUT", "error"),
        ("cancelled", "CANCELLED", "warn"),
        ("something_new", "ERROR", "error"),
    ],
)
def test_outcome_status(outcome, status, tone):
    s = OverlayState(Clock())
    command(s, outcome=outcome)
    assert tags(s)[0] == (status, tone, True)


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
    r = rows(s)
    assert r["REPLY"].text == "PLEASE REPEAT THAT"
    assert r["NOTE"].text == "NO COMMAND WORD HEARD"


def test_error_before_transcript_still_shows_what_happened():
    s = OverlayState(Clock())
    run(
        s,
        {"kind": "ptt_press", "cmd": "c1"},
        {"kind": "speak", "cmd": None, "text": "I could not understand you"},
        {"kind": "command_done", "cmd": "c1", "outcome": "not_understood"},
    )
    r = rows(s)
    assert (r["HEARD"].text, r["HEARD"].tone) == ("--", "muted")
    assert r["ACTION"].text == "NONE, NOTHING CHANGED"
    assert r["REPLY"].text == "I COULD NOT UNDERSTAND YOU"
    assert (s.history[-1]["tag"], s.history[-1]["text"]) == ("MISS", "I COULD NOT UNDERSTAND YOU")


def test_not_heard_keeps_what_was_heard_in_the_history():
    s = OverlayState(Clock())
    command(s, text="parking the more", intent="None", slots={}, outcome="not_understood")
    assert s.history[-1]["text"] == '"PARKING THE MORE"'


def test_a_key_tap_leaves_no_history():
    s = OverlayState(Clock())
    run(
        s,
        {"kind": "ptt_press", "cmd": "c1"},
        {"kind": "command_done", "cmd": "c1", "outcome": "too_short"},
    )
    assert s.history == []


def test_events_for_other_commands_ignored():
    s = OverlayState(Clock())
    command(s, cmd="c2")
    s.on_event({"kind": "stage", "stage": "stt", "cmd": "c1", "text": "old", "backend": "vosk"})
    assert s.heard == "heading one eighty"


def test_spoken_notice_when_idle():
    clock = Clock()
    s = OverlayState(clock)
    s.on_event({"kind": "speak", "cmd": None, "text": "Engine 1 did not start"})
    assert s.visible and tags(s) == [("VIMAAN", "muted", False)]
    assert s.panel().rows[0].text == "ENGINE 1 DID NOT START"
    clock.now += HOLD_SEC + 1
    assert not s.visible  # a notice is not a command: no history


def test_a_callout_is_amber():
    s = OverlayState(Clock())
    s.on_event({"kind": "callout", "callout": "flap_speed", "said": "Flap speed"})
    s.on_event({"kind": "speak", "cmd": None, "text": "Flap speed"})
    p = s.panel()
    assert tags(s) == [("CALLOUT", "warn", True)]
    assert [(r.text, r.tone) for r in p.rows] == [("FLAP SPEED", "warn")]


def test_modes():
    clock = Clock()
    s = OverlayState(clock)
    assert s.cycle_mode() == "pinned" and s.visible  # pinned shows even when idle
    p = s.panel()
    assert p.pinned and tags(s) == [("VIMAAN", "muted", False)]
    assert p.title == "HOLD THE TALK KEY"
    assert s.cycle_mode() == "off"
    command(s)
    assert not s.visible
    assert s.cycle_mode() == "auto"
    assert OverlayState(clock, mode="bogus").mode == "auto"


def test_idle_header_shows_the_wake_word_switches_and_the_phase():
    s = OverlayState(Clock(), mode="pinned")
    run(
        s,
        {"kind": "wake_word_setting", "on": True, "available": True},
        {"kind": "callouts_setting", "enabled": True},
        {"kind": "checklist_suggestions_setting", "enabled": True},
        {"kind": "flight_phase", "phase": "approach", "previous": "descent"},
    )
    p = s.panel()
    assert [t.text for t in p.tags] == ["WAKE", "CALLOUTS", "SUGGEST"]
    assert (p.title, p.meta) == ("HEY COPILOT", "APPROACH")
    s.on_event({"kind": "wake_word_setting", "on": False})
    assert s.panel().title == "HOLD THE TALK KEY"


def test_notice_does_not_interrupt_a_recording():
    s = OverlayState(Clock())
    s.on_event({"kind": "ptt_press", "cmd": "c1"})
    s.notice("Overlay pinned")
    assert s.phase == "listening"


def test_describe_slots():
    assert describe_slots({"com_port": "1", "frequency": "118.75"}) == "COM 1  freq 118.75"
    assert describe_slots({"state": "on"}) == "ON"


@pytest.mark.parametrize(
    ("intent", "slots", "text"),
    [
        ("set_autopilot_heading", {"degrees": "270"}, "HDG 270"),
        ("set_autopilot_altitude", {"altitude": "6000"}, "ALT 6,000 FT"),
        ("toggle_lights", {"light": "landing", "switch": "on"}, "LANDING LIGHTS ON"),
        ("set_com_frequency", {"com_port": "1", "frequency": "118.75"}, "COM1 118.75"),
        ("toggle_autopilot", {"state": "on"}, "AP ON"),
        ("set_autobrake", {"autobrake": "r t o"}, "AUTO BRK RTO"),
        ("set_flaps", {"position": "up"}, "FLAPS UP"),
        ("run_checklist", {"checklist": "before landing"}, "BEFORE LANDING CHECKLIST"),
        ("ask_time", {}, "TIME QUESTION"),
    ],
)
def test_action_text(intent, slots, text):
    assert action_text(intent, slots) == text


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
    r = rows(s)
    assert r["ACTION"].text == "ALT 1,000 FT" and "CONF" not in r
    assert tags(s)[0][0] == "DONE"


def test_a_to_reading_question_counts_down_then_expires():
    clock = Clock()
    s = OverlayState(clock, answer_sec=12.0)
    run(
        s,
        {"kind": "ptt_press", "cmd": "c1"},
        {"kind": "ptt_release", "cmd": "c1"},
        {"kind": "stage", "stage": "stt", "cmd": "c1", "text": "squawk two one two zero zero"},
        {
            "kind": "stage",
            "stage": "nlu",
            "cmd": "c1",
            "intent": "set_transponder_code",
            "slots": {"squawk": "21200"},
            "confidence": 1.0,
        },
        {
            "kind": "to_reading",
            "cmd": "c1",
            "intent": "set_transponder_code",
            "slots": {"squawk": "1200"},
            "repeat": False,
        },
        {"kind": "speak", "cmd": "c1", "text": "Did you mean squawk one two zero zero?"},
        {"kind": "command_done", "cmd": "c1", "outcome": "ambiguous"},
    )
    p = s.panel()
    assert tags(s)[0] == ("WHICH?", "warn", True) and p.alert == "warn" and p.meta == "12 S"
    r = rows(s)
    assert r["ACTION"].text == "SQUAWK 1200" and "21200" not in r["ACTION"].text
    assert r["ASK"].text == "DID YOU MEAN SQUAWK ONE TWO ZERO ZERO?"
    assert r["NOTE"].text == 'A "TWO" TAKEN AS "TO"'
    assert [x.value for x in p.rows if x.kind == "bar"] == [1.0]
    clock.now += 11.5  # past HOLD_SEC: a question waiting never fades
    assert s.visible and s.panel().meta == "1 S"
    clock.now += 1
    assert tags(s)[0] == ("NO ANSWER", "muted", True) and not s.panel().alert
    clock.now += HOLD_SEC + 0.1
    assert s.panel().tags[0].text != "NO ANSWER"  # gone: the idle view


def test_question_reason_shown():
    s = OverlayState(Clock())
    run(
        s,
        {"kind": "ptt_press", "cmd": "c1"},
        {"kind": "implausible", "cmd": "c1", "reason": "question"},
        {"kind": "speak", "cmd": "c1", "text": "I can't answer questions yet"},
        {"kind": "command_done", "cmd": "c1", "outcome": "non_actionable"},
    )
    assert rows(s)["NOTE"].text == "QUESTIONS AREN'T ANSWERED YET"


def test_an_answered_question_shows_the_answer():
    s = OverlayState(Clock())
    run(
        s,
        {"kind": "ptt_press", "cmd": "c1"},
        {"kind": "ptt_release", "cmd": "c1"},
        {"kind": "stage", "stage": "stt", "cmd": "c1", "text": "heading", "backend": "vosk"},
        {
            "kind": "stage",
            "stage": "nlu",
            "cmd": "c1",
            "intent": "set_autopilot_mode",
            "slots": {"mode": "heading"},
            "confidence": 1.0,
        },
        {
            "kind": "question_answered",
            "cmd": "c1",
            "intent": "ask_flight_data",
            "slots": {"reading": "heading"},
            "answer": "Heading zero niner five",
        },
        {"kind": "speak", "cmd": "c1", "text": "Heading zero niner five"},
        {"kind": "command_done", "cmd": "c1", "outcome": "answered"},
    )
    r = rows(s)
    assert tags(s)[0][0] == "ANSWER" and r["ANSWER"].text == "HEADING ZERO NINER FIVE"
    assert "ACTION" not in r and "CONF" not in r
    assert (s.history[-1]["tag"], s.history[-1]["text"]) == ("ANS", "HEADING ZERO NINER FIVE")


def test_bare_number_reason_shown():
    s = OverlayState(Clock())
    run(
        s,
        {"kind": "ptt_press", "cmd": "c1"},
        {"kind": "implausible", "cmd": "c1", "reason": "not_offered"},
        {"kind": "speak", "cmd": "c1", "text": "Say it with the word heading"},
        {"kind": "command_done", "cmd": "c1", "outcome": "rejected"},
    )
    r = rows(s)
    assert r["REPLY"].text == "SAY IT WITH THE WORD HEADING"
    assert r["NOTE"].text == "NOT ONE OF THE TWO VALUES OFFERED"


def hold_gear_up(s, cmd="c1"):
    run(
        s,
        {"kind": "ptt_press", "cmd": cmd},
        {"kind": "ptt_release", "cmd": cmd},
        {"kind": "stage", "stage": "stt", "cmd": cmd, "text": "gear up"},
        {
            "kind": "stage",
            "stage": "nlu",
            "cmd": cmd,
            "intent": "toggle_landing_gear",
            "slots": {"state": "up"},
            "confidence": 1.0,
        },
        {
            "kind": "safety_hold",
            "cmd": cmd,
            "rule": "gear_up_on_ground",
            "message": "Gear up on the ground",
        },
        {"kind": "speak", "cmd": cmd, "text": "Gear up on the ground. Say override, or repeat."},
        {"kind": "command_done", "cmd": cmd, "outcome": "safety_hold"},
    )


def test_a_safety_hold_counts_down_and_never_fades_then_expires():
    clock = Clock()
    s = OverlayState(clock, answer_sec=12.0)
    hold_gear_up(s)
    p = s.panel()
    assert tags(s)[0] == ("CAUTION", "warn", True) and p.alert == "warn"
    r = rows(s)
    assert (r["ACTION"].text, r["ACTION"].tone, r["ACTION"].detail) == ("GEAR UP", "text", "")
    assert (r["HOLD"].text, r["HOLD"].tone) == ("GEAR UP ON THE GROUND", "warn")
    assert r["SAY"].text == "OVERRIDE OR REPEAT"
    assert p.history == ()  # it needs the pilot: nothing else under it
    clock.now += 5
    p = s.panel()
    assert p.meta == "7 S"
    assert [x.value for x in p.rows if x.kind == "bar"] == [pytest.approx(7 / 12)]
    clock.now += 6.9  # past HOLD_SEC
    assert s.visible and tags(s)[0][0] == "CAUTION"
    clock.now += 0.2
    assert tags(s)[0] == ("EXPIRED", "muted", True)
    assert [x.text for x in s.panel().rows if x.kind == "note"] == [
        "NOT DONE: CONFIRMATION EXPIRED"
    ]
    assert (s.history[-1]["tag"], s.history[-1]["text"]) == ("HOLD", "GEAR UP, EXPIRED")


def test_an_override_shows_the_held_command_done():
    s = OverlayState(Clock())
    hold_gear_up(s)
    run(
        s,
        {"kind": "ptt_press", "cmd": "c2"},
        {"kind": "stage", "stage": "stt", "cmd": "c2", "text": "override"},
        {
            "kind": "safety_override",
            "cmd": "c2",
            "held_cmd": "c1",
            "intent": "toggle_landing_gear",
            "slots": {"state": "up"},
        },
        {"kind": "command_done", "cmd": "c2", "outcome": "executed"},
    )
    assert tags(s)[0][0] == "DONE" and rows(s)["ACTION"].text == "GEAR UP"
    assert [h.text for h in s.panel().history] == ["GEAR UP"]  # the hold, under the override


def test_working_until_the_plugin_says_it_is_done():
    clock = Clock()
    busy = {"c1"}
    s = OverlayState(clock, working=lambda: busy)
    command(
        s,
        cmd="c1",
        text="landing lights on",
        intent="toggle_lights",
        slots={"light": "landing", "switch": "on"},
        outcome="pending",
        reply="Landing lights on",
    )
    p = s.panel()
    assert tags(s)[0] == ("WORKING", "accent", True)
    assert [x.kind for x in p.rows if not x.label] == ["spinner"]
    assert p.history == () and p.meta == ""
    clock.now += HOLD_SEC + 30  # a slow switch: still on screen
    assert tags(s)[0][0] == "WORKING"
    busy.clear()
    assert tags(s)[0] == ("DONE", "ok", True)
    assert "REPLY" not in rows(s)  # "Landing lights on": what ACTION says
    clock.now += HOLD_SEC + 0.1
    assert s.panel().tags[0].text != "DONE"
    assert s.history[-1]["tag"] == "DONE"


def test_working_never_holds_the_screen_forever():
    clock = Clock()
    s = OverlayState(clock, working=lambda: {"c1"})
    command(s, cmd="c1", outcome="pending")
    clock.now += WORKING_MAX_SEC
    assert tags(s)[0][0] == "DONE"


def test_a_failed_step_is_red():
    s = OverlayState(Clock())  # no plugin: the end event says when
    command(s, cmd="c1", outcome="pending")
    assert tags(s)[0][0] == "WORKING"
    s.on_event({"kind": "knob_step_end", "cmd": "c1", "ok": False, "why": "stuck"})
    assert tags(s)[0] == ("FAILED", "error", True) and s.panel().alert == "error"
    assert s.history[-1]["tag"] == "FAIL"


def test_an_engine_start_ends_on_its_result():
    s = OverlayState(Clock())
    command(
        s,
        cmd="c1",
        intent="toggle_engine",
        slots={"instance": "1", "state": "on"},
        outcome="pending",
        reply="Starting engine 1",
    )
    assert rows(s)["REPLY"].text == "STARTING ENGINE 1"  # news: shown
    s.on_event({"kind": "crank_end", "cmd": "c1", "result": "running"})
    assert tags(s)[0][0] == "DONE"


def test_cancel_names_what_was_dropped():
    s = OverlayState(Clock())
    hold_gear_up(s)
    run(
        s,
        {"kind": "ptt_press", "cmd": "c2"},
        {"kind": "cancelled", "cmd": "c2", "dropped": ["toggle_landing_gear"]},
        {"kind": "speak", "cmd": "c2", "text": "Cancelled"},
        {"kind": "command_done", "cmd": "c2", "outcome": "cancelled"},
    )
    assert (s.history[-1]["tag"], s.history[-1]["text"]) == ("CNX", "GEAR")


# ---------------------------------------------------------------------------
# A checklist's own view
# ---------------------------------------------------------------------------
ITEMS = ["gear", "flaps", "speed brakes", "landing lights"]


def start_checklist(s, clock):
    run(
        s,
        {"kind": "ptt_press", "cmd": "c1"},
        {"kind": "stage", "stage": "stt", "cmd": "c1", "text": "before landing checklist"},
        {
            "kind": "stage",
            "stage": "nlu",
            "cmd": "c1",
            "intent": "run_checklist",
            "slots": {"checklist": "before landing"},
            "confidence": 1.0,
        },
        {"kind": "checklist_start", "cmd": "c1", "checklist": "before landing", "items": ITEMS},
        {"kind": "speak", "cmd": "c1", "text": "Before landing checklist"},
        {"kind": "command_done", "cmd": "c1", "outcome": "executed"},
    )
    clock.now += CHECKLIST_RESULT_SEC + 0.1


def item(s, name, state, result, ask=None):
    said = f"{name} {state}"
    run(
        s,
        {
            "kind": "checklist_item",
            "checklist": "before landing",
            "item": name,
            "state": state,
            "result": result,
            "said": said,
            "ask": ask,
        },
        {"kind": "speak", "cmd": "c1", "text": said},
    )


def test_a_checklist_shows_each_item_as_it_is_read():
    clock = Clock()
    s = OverlayState(clock)
    start_checklist(s, clock)
    p = s.panel()
    assert [t.text for t in p.tags] == ["CHKL"] and p.title == "BEFORE LANDING"
    assert p.meta == "0/4" and [r.detail for r in p.rows] == ["--"] * 4
    item(s, "gear", "down", "checked")
    item(s, "flaps", "up", "open")
    item(s, "speed brakes", "down", "offered", ask="Set speed brakes armed")
    p = s.panel()
    assert p.meta == "3/4" and p.tags[0].text == "CHKL"  # the item speech isn't a notice
    gear, flaps, brakes, lights = p.rows
    assert (gear.label, gear.detail, gear.tone) == ("GEAR", "DOWN", "ok")
    assert (flaps.detail, flaps.tone) == ("UP, NOT SET", "warn")
    assert (brakes.kind, brakes.detail) == ("offer", "DOWN")
    assert brakes.text == "SET SPEED BRAKES ARMED? SAY CONFIRM / CANCEL"
    assert (lights.detail, lights.tone) == ("--", "muted")
    clock.now += 60  # the offer waits: the list stays
    assert s.visible and s.panel().tags[0].text == "CHKL"


def test_a_checklist_offer_confirmed_then_the_checklist_ends():
    clock = Clock()
    s = OverlayState(clock)
    start_checklist(s, clock)
    item(s, "speed brakes", "down", "offered", ask="Set speed brakes armed")
    run(
        s,
        {"kind": "ptt_press", "cmd": "c2"},
        {"kind": "stage", "stage": "stt", "cmd": "c2", "text": "confirm"},
        {
            "kind": "stage",
            "stage": "nlu",
            "cmd": "c2",
            "intent": "confirm",
            "slots": {},
            "confidence": 1.0,
        },
        {
            "kind": "checklist_confirmed",
            "cmd": "c2",
            "checklist": "before landing",
            "item": "speed brakes",
            "intent": "set_speed_brake",
            "slots": {"speedbrake": "armed"},
        },
        {"kind": "speak", "cmd": "c2", "text": "Speed brakes armed"},
        {"kind": "command_done", "cmd": "c2", "outcome": "executed"},
    )
    assert rows(s)["ACTION"].text == "SPD BRK ARMED"
    clock.now += CHECKLIST_RESULT_SEC + 0.1  # back to the list
    run(
        s,
        {
            "kind": "checklist_offer",
            "checklist": "before landing",
            "item": "speed brakes",
            "result": "set",
        },
        {
            "kind": "checklist_end",
            "cmd": "c1",
            "checklist": "before landing",
            "reason": "done",
            "open": ["flaps"],
        },
        {"kind": "speak", "cmd": "c1", "text": "Before landing checklist done, one item open"},
    )
    p = s.panel()
    assert p.tags[0].text == "CHKL"
    assert [r.detail for r in p.rows if r.label == "SPEED BRAKES"] == ["SET"]
    assert [(r.text, r.tone) for r in p.rows if r.kind == "note"] == [("1 OPEN", "warn")]
    assert (s.history[-1]["tag"], s.history[-1]["text"]) == ("CHKL", "BEFORE LANDING, 1 OPEN")
    clock.now += CHECKLIST_HOLD_SEC + 0.1
    p = s.panel()
    assert p.tags[0].text != "CHKL" and p.history[0].text == "BEFORE LANDING, 1 OPEN"


def test_the_fallback_renderer_gets_flat_lines():
    s = OverlayState(Clock())
    command(s)
    assert [(ln.badge, ln.text) for ln in s.lines()] == [
        ("DONE", ""),
        ("HEARD", "heading one eighty"),
        ("ACTION", "HDG 180"),
        ("CONF", "100%"),
    ]
    assert s.lines()[0].detail == "331 MS"


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
    assert s.lines()[0].badge == "DONE"
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


def test_a_resumed_probe_starts_part_way():
    s = OverlayState(Clock())
    s.on_event(
        {
            "kind": "probe_start",
            "where": "air",
            "steps": 15,
            "start": 9,
            "name": "B738",
            "next": "V NAV",
        }
    )
    lines = s.lines()
    assert lines[0].detail == "9 / 15" and lines[-1].text.startswith("V NAV")


def test_probe_wait_and_note_lines():
    s = OverlayState(Clock())
    start_probe(s)
    s.on_event(
        {"kind": "probe_blocked", "label": "heading mode", "message": "Turn the autopilot on first"}
    )
    assert [(ln.badge, ln.text) for ln in s.lines()[1:]] == [
        ("WAIT", "Turn the autopilot on first"),
        ("NEXT", "heading mode  (press the probe key)"),
    ]
    s.on_event({"kind": "probe_act", "label": "altitude dial", "index": 14, "total": 15})
    s.on_event(
        {
            "kind": "probe_step",
            "label": "altitude dial",
            "index": 14,
            "verdict": "works",
            "next": None,
            "note": "altitude arm stayed on, which is harmless",
        }
    )
    assert [ln.badge for ln in s.lines()] == ["PROBE", "WORKS", "NOTE"]
    assert s.lines()[2].text == "Altitude arm stayed on, which is harmless"


def test_probe_summary_counts_every_saved_run():
    s = OverlayState(Clock())
    start_probe(s)
    s.on_event({"kind": "probe_end", "reason": "complete", "counts": {"works": 12, "no_change": 3}})
    assert s.lines()[-1].text == "Results saved. 12 work, 3 no change."
