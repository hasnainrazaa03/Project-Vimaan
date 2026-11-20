"""Scoring rules of scripts/model_gate.py, with a fake model (no torch)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import model_gate  # noqa: E402

COM = "set_com_frequency"


def fake(slots):
    return lambda text: {
        "intent": COM,
        "slots": dict(slots),
        "confidence": 0.99,
        "normalized_text": text,
    }


def f1(run, row):
    return model_gate.score_rows(run, [row])[COM]["slot_f1"]


def test_no_port_spoken_scores_as_com_1():
    row = {"intent": COM, "text": "frequency 121.5", "slots": {"frequency": "121.5"}}
    assert f1(fake({"frequency": "121.5"}), row) == 1.0
    want = model_gate.canonical_expected(row)[1]
    assert model_gate.judge(fake({"frequency": "121.5"}), row["text"], COM, want)[0] == "right"


def test_explicit_com_two_tuned_as_com_1_stays_wrong():
    row = {
        "intent": COM,
        "text": "frequency 133.1 on com 2",
        "slots": {"frequency": "133.1", "com_port": "2"},
    }
    want = model_gate.canonical_expected(row)[1]
    assert want["com_port"] == "2"
    missed = fake({"frequency": "133.1"})  # the model didn't tag the port
    assert f1(missed, row) < 1.0
    assert model_gate.judge(missed, row["text"], COM, want)[0] == "wrong"
    tagged = fake({"frequency": "133.1", "com_port": "2"})
    assert f1(tagged, row) == 1.0
    assert model_gate.judge(tagged, row["text"], COM, want)[0] == "right"


def test_gate_counts_an_ambiguous_target_as_not_acted_on():
    p = {
        "intent": "set_autopilot_altitude",
        "slots": {"altitude": "21000"},
        "confidence": 0.99,
        "normalized_text": "set altitude 21000 feet",
        "original_text": "set altitude two one thousand feet",
    }
    assert model_gate.decision(p) == ("ambiguous_two", {})
    p["original_text"] = "set altitude twenty one thousand feet"
    assert model_gate.decision(p)[0] == "act"


def test_gate_counts_a_clipped_number_as_not_acted_on():
    # As the plugin: "altitude thousand" lost the start of the number.
    p = {
        "intent": "set_autopilot_altitude",
        "slots": {"altitude": "1000"},
        "confidence": 0.99,
        "normalized_text": "altitude 1000",
        "original_text": "altitude thousand",
    }
    assert model_gate.decision(p) == ("clipped_number", {})
    p["original_text"] = "altitude one thousand"
    assert model_gate.decision(p)[0] == "act"


def test_gate_counts_a_lost_digit_as_not_acted_on():
    # As the plugin: "one seven zero" heard as "want seven zero".
    p = {
        "intent": "set_autopilot_heading",
        "slots": {"degrees": "70"},
        "confidence": 0.99,
        "normalized_text": "turn left heading want 70",
        "original_text": "turn left heading want seven zero",
    }
    assert model_gate.decision(p) == ("lost_digits", {})
    p["original_text"] = "turn left heading seven zero"
    assert model_gate.decision(p)[0] == "act"


def test_gate_counts_a_question_as_not_acted_on():
    # As the plugin: v26 read "is the autopilot off" as "autopilot off".
    p = {
        "intent": "toggle_autopilot",
        "slots": {"state": "off", "instance": "1"},
        "confidence": 0.99,
        "normalized_text": "is the autopilot off",
        "original_text": "is the autopilot off",
    }
    assert model_gate.decision(p) == ("question", {})
    p["original_text"] = p["normalized_text"] = "autopilot off"
    assert model_gate.decision(p)[0] == "act"


def _q(intent, conf=0.99, text="what's our altitude", slots=None):
    return {"intent": intent, "slots": slots or {}, "confidence": conf, "normalized_text": text}


def test_a_question_model_answers_only_a_sure_question_with_its_words():
    assert model_gate.answer_decision(_q("ask_flight_data")) == "answer"
    assert model_gate.answer_decision(_q("ask_flight_data", conf=0.3)) == "low_confidence"
    assert (
        model_gate.answer_decision(_q("ask_fuel", text="what's our altitude")) == "no_command_word"
    )
    assert model_gate.answer_decision(_q("None")) == "not_a_question"
    assert model_gate.answer_decision(_q("set_flaps", text="flaps five")) == "not_a_question"


def test_a_question_model_never_answers_a_word_alone():
    assert model_gate.answer_decision(_q("ask_autopilot", text="armed")) == "one_word"
    assert model_gate.answer_decision(_q("ask_autopilot", text="uh armed")) == "one_word"
    assert model_gate.answer_decision(_q("ask_autopilot", text="what's armed")) == "answer"


def test_question_dev_score_counts_right_misanswered_and_answered_commands():
    from utils.dev_score import question_score

    table = {
        "what's our altitude": _q("ask_flight_data", slots={"reading": "altitude"}),
        "what's the heading": _q(
            "ask_flight_data", text="what's the heading", slots={"reading": "altitude"}
        ),
        "how much fuel": _q("ask_fuel", conf=0.2, text="how much fuel"),
        "gear down": _q("ask_switch", text="gear down", slots={"switch_system": "gear"}),
        "flaps five": _q("None", text="flaps five"),
    }
    rows = [
        {
            "text": "what's our altitude",
            "intent": "ask_flight_data",
            "slots": {"reading": "altitude"},
        },
        {
            "text": "what's the heading",
            "intent": "ask_flight_data",
            "slots": {"reading": "heading"},
        },
        {"text": "how much fuel", "intent": "ask_fuel", "slots": {}},
        {"text": "gear down", "intent": "None", "slots": {}},
        {"text": "flaps five", "intent": "None", "slots": {}},
    ]
    s = question_score(lambda t: table[t], rows, wrong_weight=5.0)
    assert (s["right"], s["misanswered"], s["declined"]) == (1, 1, 1)
    assert (s["others"], s["others_answered"]) == (2, 1)
    assert s["score"] == 1 - 1 - 5.0


def test_question_dev_score_weighs_every_phrasing_the_same():
    # q1 attempt 2: one dev phrasing with many rows chose the run.
    from utils.dev_score import PHRASING_POINTS, question_score

    right = _q("ask_flight_data", slots={"reading": "altitude"})
    wrong = _q("ask_flight_data", slots={"reading": "heading"})
    row = {"text": "", "intent": "ask_flight_data", "slots": {"reading": "altitude"}}
    rows = [{**row, "text": f"big {i}", "template": "ask_flight_data#1"} for i in range(9)]
    rows += [{**row, "text": "small", "template": "ask_flight_data#2"}]
    # The big phrasing wrong, the small one right: one phrasing each way.
    s = question_score(lambda t: wrong if t.startswith("big") else right, rows)
    assert (s["phrasings"], s["right"], s["misanswered"]) == (2, 1, 9)
    assert s["score"] == 0
    s = question_score(lambda t: right, rows)
    assert s["score"] == 2 * PHRASING_POINTS


def _p(intent, slots, text):
    return {
        "intent": intent,
        "slots": dict(slots),
        "confidence": 0.99,
        "normalized_text": text,
        "original_text": text,
    }


def test_gate_counts_a_target_or_change_as_asked():
    # Phase L.1: the plugin asks unless the aircraft rules a reading out.
    bare = _p("set_autopilot_altitude", {"altitude": "2000"}, "descend two thousand feet")
    assert model_gate.decision(bare) == ("ambiguous_by", {})
    turn = _p("set_autopilot_heading", {"degrees": "120"}, "turn right one two zero")
    assert model_gate.decision(turn) == ("ambiguous_by", {})
    marked = _p("set_autopilot_altitude", {"altitude": "2000"}, "descend to two thousand feet")
    assert model_gate.decision(marked)[0] == "act"


def test_gate_scores_a_turn_flown_as_a_heading_as_that_heading():
    p = _p("turn_by", {"turn": "right", "turn_degrees": "270"}, "turn right two seven zero")
    assert model_gate.decide(p) == ("act", "set_autopilot_heading", {"degrees": "270"})
    run = lambda text: p  # noqa: E731
    want = {"degrees": "270"}
    assert model_gate.judge(run, p["original_text"], "set_autopilot_heading", want)[0] == "right"


def test_gate_scores_a_relative_command():
    p = _p("turn_by", {"turn": "left", "turn_degrees": "20"}, "turn left twenty degrees")
    assert model_gate.decide(p) == ("act", "turn_by", {"turn": "left", "turn_degrees": "20"})
    low = _p("set_autopilot_altitude", {"altitude": "500"}, "descend five hundred feet")
    assert model_gate.decide(low) == (
        "act",
        "change_altitude_by",
        {"direction": "down", "altitude_change": "500"},
    )
    north = _p("set_compass_heading", {"compass": "west"}, "turn west twenty")
    assert model_gate.decision(north) == ("compass_conflict", {})
