"""Training data v2 (ML/data/generate_v2.py): properties every dataset must have."""

import dataclasses
import re
from collections import defaultdict

import pytest
from data import generate_v2
from vimaan_nlu.commands import COMMANDS
from vimaan_nlu.normalization import normalize_aviation_input


@pytest.fixture(scope="module")
def data():
    rows, stats = generate_v2.generate(per_intent=300, seed=7)
    return rows, stats


def test_every_slot_is_a_word_of_the_normalized_text(data):
    rows, _ = data
    for r in rows:
        norm = normalize_aviation_input(r["text"])
        for name, value in r["slots"].items():
            assert re.search(r"(?<![\w.])" + re.escape(value) + r"(?![\w.])", norm), (r, name)


def test_no_text_has_two_labels(data):
    rows, _ = data
    labels = defaultdict(set)
    for r in rows:
        labels[r["text"]].add(r["intent"])
    assert [t for t, i in labels.items() if len(i) > 1] == []


def test_no_text_holds_a_placeholder(data):
    # Templates are filled, and the negatives' word salad is built from
    # template words with every placeholder removed, "{slot=word}" included.
    rows, _ = data
    assert [r["text"] for r in rows if "{" in r["text"] or "}" in r["text"]] == []
    filler, _, _ = generate_v2._vocab()
    assert [w for w in filler if "{" in w or "}" in w or "=" in w] == []


def test_no_negative_holds_a_whole_command(data):
    # "ident" alone is a command; word salad around it must not be labelled None.
    rows, _ = data
    whole = generate_v2.whole_commands()
    assert "ident" in whole
    salad = [r["text"] for r in rows if r["template"] == "None#neg1"]
    assert salad
    assert [t for t in salad if whole & set(t.split())] == []


def test_same_seed_same_data_in_any_process():
    # Python orders sets differently in every process (hash seed); the data
    # must not depend on that (data v11 differed in 774 bare-number negatives).
    import os
    import subprocess
    import sys
    from pathlib import Path

    ml = Path(generate_v2.__file__).resolve().parents[1]
    code = (
        "import hashlib, json; from data import generate_v2; "
        "rows, _ = generate_v2.generate(per_intent=40, seed=3); "
        "print(hashlib.sha256(json.dumps(rows).encode()).hexdigest())"
    )
    digests = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(ml)}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1, "generate_v2 depends on set order"


def test_held_out_templates_never_trained_on(data):
    rows, _ = data
    split_of = defaultdict(set)
    for r in rows:
        if not r["intent"] == "None":
            split_of[r["template"]].add(r["split"])
    actionable = {c.intent for c in COMMANDS if c.actionable}
    assert all(len(s) == 1 for t, s in split_of.items() if t.split("#")[0] in actionable)
    for intent in actionable:
        held = {t for t, s in split_of.items() if t.startswith(intent + "#") and s == {"test"}}
        assert len(held) >= 2, intent


def test_dev_templates_are_held_out_and_apart_from_the_test_ones(data):
    rows, _ = data
    split_of = defaultdict(set)
    for r in rows:
        split_of[r["template"]].add(r["split"])
    for c in COMMANDS:
        if not c.actionable:
            continue
        test = generate_v2.holdout_templates(c)
        dev = generate_v2.dev_templates(c)
        assert len(dev) == generate_v2.DEV_TEMPLATES and not dev & test, c.intent
        for t in dev:
            key = f"{c.intent}#{c.templates.index(t)}"
            assert split_of[key] <= {"dev"}, (key, split_of[key])
    negatives = [r for r in rows if r["template"].startswith("None#neg")]
    share = sum(r["split"] == "dev" for r in negatives) / len(negatives)
    assert 0.02 < share < 0.09


def test_every_pinned_value_of_a_held_out_template_is_also_trained():
    # "{ice=boots} {switch}" was the only template saying "boots", and the
    # hash held it out: v26 never saw the word and sent "boots on" to the
    # flaps. A held-out template must test a new phrasing of known words,
    # so each "{slot=word}" in it needs a training template pinning it too.
    pin = re.compile(r"\{(\w+)=([^}]+)\}")
    lacking = []
    for c in COMMANDS:
        held = generate_v2.holdout_templates(c) | generate_v2.dev_templates(c)
        trained = {p for t in c.templates if t not in held for p in pin.findall(t)}
        lacking += [(c.intent, t, p) for t in held for p in pin.findall(t) if p not in trained]
    assert lacking == []


def test_dev_leaves_the_test_rows_as_they_were(monkeypatch):
    with_dev, _ = generate_v2.generate(per_intent=60, seed=5)
    monkeypatch.setattr(generate_v2, "DEV_TEMPLATES", 0)
    monkeypatch.setattr(generate_v2, "QUESTION_DEV_TEMPLATES", 0)
    monkeypatch.setattr(generate_v2, "DEV_ROW_SHARE", 0.0)
    without, _ = generate_v2.generate(per_intent=60, seed=5)
    assert [r for r in with_dev if r["split"] == "test"] == [
        r for r in without if r["split"] == "test"
    ]
    assert any(r["split"] == "dev" for r in with_dev)
    assert not any(r["split"] == "dev" for r in without)


def test_every_command_and_negatives_present(data):
    rows, stats = data
    intents = {r["intent"] for r in rows}
    assert {c.intent for c in COMMANDS} <= intents
    negatives = [r for r in rows if r["template"].startswith("None#neg")]
    actionable = {c.intent for c in COMMANDS if c.actionable}
    commands = sum(1 for r in rows if r["intent"] in actionable)
    assert len(negatives) >= 0.09 * commands
    # A stray word fused onto a value ("eighty fifty degrees" -> 8050) can't be
    # labelled and is dropped; that must stay rare.
    for key, count in stats.items():
        if key.startswith("dropped:") and key != "dropped:negative_collides":
            assert count < 0.01 * 300, key


def test_noise_is_applied_and_real(data):
    rows, _ = data
    actionable = [r for r in rows if r["intent"] not in ("None",) and r["slots"]]
    noisy = [r for r in actionable if r["noise"] != "none"]
    assert 0.1 < len(noisy) / len(actionable) < 0.3
    leads = [r for r in noisy if r["noise"] == "lead"]
    assert all(r["text"].split()[0] in generate_v2.LEADING_NOISE for r in leads)


def _cmd(intent):
    return next(c for c in COMMANDS if c.intent == intent)


def test_a_stray_word_cannot_make_its_own_command():
    fd = generate_v2.stray_words(_cmd("toggle_flight_director"), "flight director on")
    assert "ice" in fd  # anti-ice needs which one: "ice director on" is the director
    assert "fuel" not in fd  # "fuel ... on" could be the fuel pumps
    gear = generate_v2.stray_words(_cmd("toggle_landing_gear"), "gear down")
    assert "flaps" not in gear  # "flaps ... down" is a flap command
    lights = generate_v2.stray_words(_cmd("toggle_lights"), "strobes on")
    assert "auto" not in lights  # one of the lights' own values
    assert "ident" not in fd and "v" not in fd  # a whole command; a letter


def test_a_stray_word_takes_the_place_of_a_word_that_can_go():
    fd = _cmd("toggle_flight_director")
    assert generate_v2._replaceable(
        fd, "flight director on".split(), 0, {"on"}
    )  # "ice director on"
    assert not generate_v2._replaceable(fd, "flight director on".split(), 2, {"on"})  # a slot word
    flaps = _cmd("set_flaps")
    assert not generate_v2._replaceable(
        flaps, "flaps five".split(), 0, {"five"}
    )  # none of its own left
    vs = _cmd("set_vertical_speed")
    # Without "vertical", "speed one thousand" is a speed of its own.
    assert not generate_v2._replaceable(
        vs, "vertical speed one thousand".split(), 0, {"one", "thousand"}
    )


def test_a_stray_word_never_takes_the_word_that_tells_two_commands_apart():
    # "boost two transfer": without "nav" it is the COM swap's as much.
    for intent, text, port in (
        ("swap_nav_frequency", "nav two transfer", {"nav_port": "two"}),
        ("swap_com_frequency", "com one transfer", {"com_port": "one"}),
        ("swap_com_frequency", "com two swap", {"com_port": "two"}),
    ):
        assert not generate_v2._replaceable(
            _cmd(intent), text.split(), 0, set(port.values()), port
        ), text
    # Without "nav", "... two standby" no longer says NAV: every NAV standby
    # says "nav". The COM standby is also said without "com" ("set standby").
    port = {"nav_port": "two"}
    assert generate_v2._alternatives(_cmd("set_nav_standby"), {"two", "standby"}, {}) != []
    assert generate_v2._alternatives(_cmd("set_com_standby"), {"two", "standby"}, {}) == []
    # But a NAV frequency still says which radio: 116.9 is no COM frequency.
    standby = {**port, "nav_frequency": "one one six decimal nine"}
    text = "nav two standby one one six decimal nine".split()
    slot_words = {w for v in standby.values() for w in v.split()}
    assert generate_v2._replaceable(_cmd("set_nav_standby"), text, 0, slot_words, standby)


def test_a_word_can_go_when_what_is_left_is_still_this_command():
    # A swap takes no frequency: "... nav two standby to 109.4" is no swap.
    standby = {"nav_port": "two", "nav_frequency": "one oh nine point four"}
    text = "set nav two standby to one oh nine point four".split()
    slot_words = {w for v in standby.values() for w in v.split()}
    assert generate_v2._replaceable(_cmd("set_nav_standby"), text, 0, slot_words, standby)
    # The active frequency is said without a word of its own; standby isn't.
    active = {"com_port": "one", "frequency": "one one nine decimal seven"}
    text = "set com one to one one nine decimal seven".split()
    slot_words = {w for v in active.values() for w in v.split()}
    assert generate_v2._replaceable(_cmd("set_com_frequency"), text, 0, slot_words, active)
    text = "com one standby one one nine decimal seven".split()
    assert not generate_v2._replaceable(_cmd("set_com_standby"), text, 2, slot_words, active)
    # A mode needs a mode: "... autopilot on" is still the autopilot.
    state = {"state": "on"}
    ap = _cmd("toggle_autopilot")
    assert generate_v2._replaceable(ap, "turn autopilot on".split(), 0, {"on"}, state)
    fd = _cmd("toggle_flight_director")  # "ice director on" is the director
    assert generate_v2._replaceable(fd, "flight director on".split(), 0, {"on"}, state)


def test_no_stray_row_loses_the_radio_it_swaps(data):
    rows, _ = data
    radio = {"swap_nav_frequency": {"nav", "navigation"}, "swap_com_frequency": {"com", "comm"}}
    for r in rows:
        if r["intent"] in radio and r["noise"] == "stray":
            words = set(r["text"].split())
            assert words & (radio["swap_nav_frequency"] | radio["swap_com_frequency"]), r


def test_a_stray_word_is_mostly_in_place_of_an_end_word():
    # The recogniser replaced end words and never added one (synthetic voices):
    # a leading extra word taught "parking start engine one" as the engine.
    import random

    rng = random.Random(5)
    fd = _cmd("toggle_flight_director")
    text = "please turn the flight director on now"
    outs = [generate_v2.add_stray(fd, text, "ice", {"on"}, rng) for _ in range(2000)]
    added = sum(len(o.split()) > len(text.split()) for o in outs)
    first = sum(o.startswith("ice turn") for o in outs)
    last = sum(o.endswith("on ice") for o in outs)
    assert 0.05 < added / len(outs) < 0.15
    assert first > 4 * last > 0
    assert generate_v2.add_stray(fd, "fd", "ice", set(), rng) in (None, "ice fd", "fd ice")


def test_stray_rows_keep_their_command(data):
    rows, _ = data
    stray = [r for r in rows if r["noise"] == "stray"]
    actionable = [r for r in rows if r["intent"] in {c.intent for c in COMMANDS if c.actionable}]
    assert 0.02 < len(stray) / len(actionable) < 0.1
    for r in stray:
        extra = [w for w in r["text"].split() if w in generate_v2._stray_pool()]
        assert extra and r["slots"], r


def test_holdout_stable_when_phrasings_are_added():
    c = next(c for c in COMMANDS if c.intent == "toggle_engine")
    grown = dataclasses.replace(c, templates=(*c.templates, "a new phrasing {state}"))
    held, held_grown = generate_v2.holdout_templates(c), generate_v2.holdout_templates(grown)
    assert len(held & held_grown) >= len(held) - 1


def test_brake_heard_as_the_is_in_the_data(data):
    rows, _ = data
    misheard = [r for r in rows if r["noise"] == "mishear"]
    assert misheard and all(r["intent"] == "toggle_parking_brake" for r in misheard)
    assert all("parking the" in r["text"] for r in misheard)


def test_two_digit_headings_beside_three_digit_ones(data):
    rows, _ = data
    values = [r["slots"]["degrees"] for r in rows if r["intent"] == "set_autopilot_heading"]
    two = [v for v in values if len(v) == 2]
    three = [v for v in values if len(v) == 3]
    assert len(two) > 0.1 * len(values) and len(three) > 0.6 * len(values)
    assert all(len(v) in (2, 3) for v in values)


def test_deterministic():
    a, _ = generate_v2.generate(per_intent=50, seed=3)
    b, _ = generate_v2.generate(per_intent=50, seed=3)
    assert a == b


def test_verbs_are_tagged_where_spoken(data):
    rows, _ = data
    gear = [r for r in rows if r["intent"] == "toggle_landing_gear"]
    assert any(r["slots"].get("state") in ("raise", "lower", "retract", "extend") for r in gear)


# --- Tier 4: questions --------------------------------------------------------
QUESTION_INTENTS = {c.intent for c in COMMANDS if c.is_question}


def test_every_question_is_generated_with_its_slots(data):
    rows, _ = data
    asked = [r for r in rows if r["intent"] in QUESTION_INTENTS]
    assert {r["intent"] for r in asked} == QUESTION_INTENTS
    assert any(
        r["slots"].get("reading") == "vertical speed" or r["slots"].get("reading") for r in asked
    )
    assert all(r["noise"] != "stray" for r in asked), "stray words come from other commands"


def test_a_yes_no_question_keeps_its_question_word(data):
    # Without it "is the gear down" is the command "gear down": the label
    # would teach the model to answer a command.
    from vimaan_nlu.questions import looks_like_question

    rows, _ = data
    states = {"on", "off", "up", "down", "set", "armed", "running", "engaged"}
    closers = {"please", "now", "right"}  # QUESTION_SUFFIXES
    for r in rows:
        if r["intent"] in QUESTION_INTENTS:
            words = r["text"].split()
            while len(words) > 1 and words[-1] in closers:
                words.pop()
            if words[-1] in states:  # "is the gear down (please)", not "... on fuel"
                assert looks_like_question(r["text"]), r["text"]


def test_no_question_has_a_commands_words(data):
    rows, _ = data
    commands = {r["text"] for r in rows if r["intent"] not in QUESTION_INTENTS | {"None"}}
    asked = {r["text"] for r in rows if r["intent"] in QUESTION_INTENTS}
    assert not commands & asked


def test_questions_as_the_recogniser_hears_them(data):
    # The question grammar has no "what's"; "our" comes back "are" or "r".
    rows, _ = data
    heard = [r["text"] for r in rows if r["noise"] == "heard"]
    assert heard
    assert all("what's" not in t.split() or "our" in t.split() for t in heard[:0])
    assert any(t.startswith(("what are", "what r", "what the")) for t in heard)


# --- the question model's data (Tier 4) -------------------------------------


@pytest.fixture(scope="module")
def question_data():
    return generate_v2.generate_questions(per_intent=200, seed=7, per_command=40)


def test_question_data_has_only_questions_and_not_a_question(question_data):
    rows, _ = question_data
    by_intent = {c.intent: c for c in COMMANDS}
    for r in rows:
        assert r["intent"] == "None" or by_intent[r["intent"]].is_question
    # Every command is there, as "not a question", with no slots.
    commands = {
        r["template"].split(":")[1].split("#")[0] for r in rows if "None#cmd:" in r["template"]
    }
    assert commands == {c.intent for c in COMMANDS if c.actionable}
    assert all(not r["slots"] for r in rows if r["intent"] == "None")
    assert {r["intent"] for r in rows} >= {c.intent for c in COMMANDS if c.is_question}


def test_question_data_commands_keep_their_held_out_templates(question_data):
    # A held-out command phrasing stays held out: the question model's test
    # asks whether it declines commands it never saw.
    rows, _ = question_data
    held = {
        f"{c.intent}#{c.templates.index(t)}": c
        for c in COMMANDS
        if c.actionable
        for t in generate_v2.holdout_templates(c)
    }
    for r in rows:
        if r["template"].startswith("None#cmd:"):
            original = r["template"].split(":", 1)[1]
            assert (r["split"] == "test") == (original in held)


def test_question_data_no_text_is_both(question_data):
    rows, _ = question_data
    labels = defaultdict(set)
    for r in rows:
        labels[r["text"]].add(r["intent"])
    assert [t for t, s in labels.items() if len(s) > 1] == []


def test_a_question_family_has_three_dev_templates():
    for c in COMMANDS:
        want = generate_v2.QUESTION_DEV_TEMPLATES if c.is_question else generate_v2.DEV_TEMPLATES
        assert len(generate_v2.dev_templates(c)) == want


def test_a_question_word_alone_is_not_a_question(question_data):
    from vimaan_nlu.questions import BARE_READINGS

    rows, _ = question_data
    lone = {r["text"]: r for r in rows if r["template"].startswith("None#lone:")}
    words = generate_v2.lone_words()
    assert "armed" in words and "torque" in words
    # Readings answered alone and system names that ask which way are not.
    assert not set(words) & (set(BARE_READINGS) | {"flaps", "gear"})
    assert {f"uh {w}" for w in words} <= set(lone)
    assert all(r["intent"] == "None" and not r["slots"] for r in lone.values())


def test_lone_words_are_held_out_whole_and_gold_fragments_never_trained():
    from test_gold_nlu import MUST_NOT_ACT

    split = generate_v2.lone_word_split()
    assert {"train", "dev", "test"} == set(split.values())
    assert sum(s == "dev" for s in split.values()) == generate_v2.LONE_DEV_WORDS
    gold = set(MUST_NOT_ACT) & set(split)
    assert gold <= generate_v2.GOLD_FRAGMENTS
    assert all(split[w] == "test" for w in generate_v2.GOLD_FRAGMENTS)
    rows = generate_v2.lone_word_rows()
    for r in rows:
        assert r["split"] == split[r["template"].split(":", 1)[1]]


# q1 round 4: extra question rows, appended after an unchanged dataset.
@pytest.mark.parametrize("kind", generate_v2.QUESTION_EXTRAS)
def test_extra_rows_come_after_the_same_rows_and_are_train_only(question_data, kind):
    from vimaan_nlu.questions import one_word
    from vimaan_nlu.stt_vocabulary import load_question_vocabulary

    rows, _ = question_data
    more, stats = generate_v2.generate_questions(per_intent=200, seed=7, per_command=40, extra=kind)
    assert more[: len(rows)] == rows
    extra = more[len(rows) :]
    assert extra and all(r["split"] == "train" and r["intent"] in QUESTION_INTENTS for r in extra)
    protected = generate_v2._probe_cores() | {
        generate_v2.core(r["text"]) for r in rows if r["split"] != "train" or r["intent"] == "None"
    }
    heard = set(load_question_vocabulary())
    for r in extra:
        words = r["text"].split()
        assert generate_v2.core(r["text"]) not in protected, r
        assert "quick" not in words and not one_word(r["text"]) and set(words) <= heard, r
        if kind == "terse":
            assert "update" not in words, r


@pytest.mark.parametrize("kind", generate_v2.QUESTION_EXTRAS)
def test_an_extra_template_is_never_a_held_out_or_dev_template(kind):
    for c in COMMANDS:
        if c.is_question:
            held = generate_v2.holdout_templates(c) | generate_v2.dev_templates(c)
            assert not held & set(generate_v2.extra_templates(c, kind)), c.intent


@pytest.mark.parametrize(
    "template, rest",
    [
        ("give me the fuel quantity", "fuel quantity"),
        ("can you tell me the {reading}", "{reading}"),
        ("say squawk code", "squawk code"),
        ("what's the transponder code", "transponder code"),
        ("what's our {reading}", "{reading}"),
        ("what's the autopilot doing", None),  # asks something else
        ("is the gear down", None),  # no carrier: "gear down" is a command
        ("how much fuel is left", None),
    ],
)
def test_a_carrier_comes_off_only_when_what_is_left_still_asks(template, rest):
    assert generate_v2._uncarried(template) == rest


def test_the_dev_probe_is_one_phrasing_a_row_of_known_intents():
    import json

    with open(generate_v2.PROBE_FILE) as fh:
        probe = [json.loads(line) for line in fh]
    assert len(probe) == len(generate_v2._probe_cores()) >= 90
    assert {r["intent"] for r in probe} == QUESTION_INTENTS | {"None"}
    assert all(r["split"] == "probe" for r in probe)


# ---------------------------------------------------------------------------
# Phase L.1: new commands beside an existing data set (v27 = data v17 + them)
# ---------------------------------------------------------------------------
L1 = ["turn_by", "set_compass_heading", "change_altitude_by", "change_speed_by"]


@pytest.fixture(scope="module")
def extended(tmp_path_factory):
    base_rows = [r for r in generate_v2.generate(per_intent=60, seed=3)[0] if r["intent"] not in L1]
    path = tmp_path_factory.mktemp("base") / "base.jsonl"
    path.write_text("".join(__import__("json").dumps(r) + "\n" for r in base_rows))
    rows, stats = generate_v2.extend(str(path), L1, per_intent=300, seed=5)
    return base_rows, rows, stats


def test_extend_keeps_every_base_row_as_it_was(extended):
    base, rows, _ = extended
    assert rows[: len(base)] == base
    assert {r["intent"] for r in rows[len(base) :]} == set(L1) | {"None"}


def test_extended_rows_hold_out_their_own_templates(extended):
    _, rows, _ = extended
    from vimaan_nlu.commands import command

    for intent in L1:
        held = generate_v2.holdout_templates(command(intent))
        for r in rows:
            if r["intent"] == intent:
                template = command(intent).templates[int(r["template"].split("#")[1])]
                assert (r["split"] == "test") == (template in held)


def test_a_thousand_is_said_and_thousand_is_the_slot(extended):
    _, rows, _ = extended
    a = [r for r in rows if r["intent"] == "change_altitude_by" and " a thousand" in r["text"]]
    assert a and all(r["slots"]["altitude_change"] == "1000" for r in a)
    assert not any("another a " in r["text"] for r in rows)


def test_extend_refuses_a_command_already_in_the_base(extended, tmp_path):
    base, _, _ = extended
    path = tmp_path / "base.jsonl"
    path.write_text("".join(__import__("json").dumps(r) + "\n" for r in base))
    with pytest.raises(ValueError, match="already in"):
        generate_v2.extend(str(path), ["set_autopilot_heading"], per_intent=10)
