"""Phase L.3.1: finding two commands in one breath (vimaan_nlu.multi)."""

import json
from pathlib import Path

import pytest
from vimaan_nlu import multi

requires_torch = pytest.mark.requires_torch
FIXTURES = Path(__file__).parent / "fixtures"


def table(readings):
    """A stand-in model: text -> (intent, slots[, confidence])."""

    def predict(text):
        intent, slots, *conf = readings.get(text, ("None", {}, 0.4))
        return {"intent": intent, "slots": slots, "confidence": conf[0] if conf else 0.99}

    return predict


def test_two_whole_commands_are_found():
    predict = table(
        {
            "gear down flaps fifteen": ("set_flaps", {"position": "down", "flap_setting": "15"}),
            "gear down": ("toggle_landing_gear", {"state": "down"}),
            "flaps fifteen": ("set_flaps", {"flap_setting": "15"}),
        }
    )
    two = multi.two_commands("gear down flaps fifteen", predict)
    assert [(p["text"], p["intent"]) for p in two] == [
        ("gear down", "toggle_landing_gear"),
        ("flaps fifteen", "set_flaps"),
    ]


def test_one_command_word_is_one_command():
    calls = []
    assert multi.two_commands("gear down", lambda t: calls.append(t)) is None
    assert calls == []  # no extra model runs


def test_a_split_never_falls_inside_a_number():
    words = "descend at one thousand eight hundred feet per minute".split()
    for first, second, _ in multi.splits(words):
        assert second[0] not in multi.NUMBER_WORDS


def test_a_split_never_cuts_a_commands_own_phrase():
    # "nav standby" is one phrase of the NAV standby frequency.
    words = "confirm nav standby frequency one one three point seven".split()
    assert all((a[-1], b[0]) != ("nav", "standby") for a, b, _ in multi.splits(words))


def test_the_same_command_twice_with_the_same_settings_is_one():
    predict = table(
        {
            "set parking brake on": ("toggle_parking_brake", {"state": "on"}),
            "set parking": ("toggle_parking_brake", {"state": "on"}),
            "brake on": ("toggle_parking_brake", {"state": "on"}),
        }
    )
    assert multi.two_commands("set parking brake on", predict) is None


def test_the_same_command_for_two_lights_is_two():
    predict = table(
        {
            "landing lights on taxi lights on": (
                "toggle_lights",
                {"light": "taxi", "switch": "on"},
            ),
            "landing lights on": ("toggle_lights", {"light": "landing", "switch": "on"}),
            "taxi lights on": ("toggle_lights", {"light": "taxi", "switch": "on"}),
        }
    )
    assert multi.two_commands("landing lights on taxi lights on", predict) is not None


def test_all_of_it_read_as_a_third_command_is_one():
    # "set vertical speed | climb two thousand" is a vertical speed.
    predict = table(
        {
            "set vertical speed climb two thousand": (
                "set_vertical_speed",
                {"vs": "2000", "direction": "up"},
            ),
            "set vertical speed": ("set_autopilot_mode", {"mode": "vertical speed"}),
            "climb two thousand": ("set_autopilot_altitude", {"altitude": "2000"}),
        }
    )
    assert multi.two_commands("set vertical speed climb two thousand", predict) is None


def test_a_turn_and_a_heading_are_one_kind():
    # Read whole as a heading, "turn right thirty degrees ..." is the turn's.
    predict = table(
        {
            "turn right thirty degrees altitude four thousand": (
                "set_autopilot_heading",
                {"degrees": "30"},
            ),
            "turn right thirty degrees": ("turn_by", {"turn": "right", "turn_degrees": "30"}),
            "altitude four thousand": ("set_autopilot_altitude", {"altitude": "4000"}),
        }
    )
    assert multi.two_commands("turn right thirty degrees altitude four thousand", predict)


def test_a_command_with_on_or_off_must_say_it():
    # "alternators ahead" names the generators and nothing to do with them.
    p = {"intent": "toggle_electrical", "slots": {"power": "generator"}, "confidence": 0.99}
    assert not multi.whole(p, "alternators ahead")
    flaps = {"intent": "set_flaps", "slots": {"flap_setting": "15"}, "confidence": 0.99}
    assert multi.whole(flaps, "flaps fifteen")  # a number says what to do
    mode = {"intent": "set_autopilot_mode", "slots": {"mode": "heading"}, "confidence": 0.99}
    assert multi.whole(mode, "heading mode")  # its name says on


def test_two_names_sharing_one_verb_are_two():
    predict = table(
        {
            "gear and flaps up": ("set_flaps", {"position": "up"}),
            "gear": ("toggle_landing_gear", {}),
            "flaps up": ("set_flaps", {"position": "up"}),
            "landing and taxi lights on": ("toggle_lights", {"light": "taxi", "switch": "on"}),
            "taxi lights on": ("toggle_lights", {"light": "taxi", "switch": "on"}),
        }
    )
    assert multi.two_commands("gear and flaps up", predict)
    assert multi.two_commands("landing and taxi lights on", predict)  # "landing": another light


def test_a_go_ahead_garble_is_not_a_shared_verb():
    predict = table(
        {
            "beacon ahead and strobes on": ("toggle_lights", {"light": "strobe", "switch": "on"}),
            "beacon ahead": ("toggle_lights", {"light": "beacon"}),
            "strobes on": ("toggle_lights", {"light": "strobe", "switch": "on"}),
        }
    )
    assert multi.two_commands("beacon ahead and strobes on", predict) is None


def test_a_part_the_plugin_wouldnt_act_on_isnt_a_command():
    predict = table(
        {
            "heading two seven zero climb five thousand": (
                "set_autopilot_heading",
                {"degrees": "270"},
            ),
            "heading two seven zero": ("set_autopilot_heading", {"degrees": "270"}, 0.3),
            "climb five thousand": ("set_autopilot_altitude", {"altitude": "5000"}),
        }
    )
    assert multi.two_commands("heading two seven zero climb five thousand", predict) is None


# ---- with the installed model -------------------------------------------------
@pytest.fixture(scope="session")
def read():
    from utils import get_latest_model_path

    if not get_latest_model_path():
        pytest.skip("no trained model present")
    from vimaan_nlu import predict
    from vimaan_nlu.model_loader import ModelLoader

    loader = ModelLoader()
    loader.load_all()
    if "turn_by" not in set(loader.intent_map_rev.values()):
        pytest.skip("the installed model predates relative commands (phase L.1)")
    return lambda text: predict(
        text,
        loader.model,
        loader.tokenizer,
        loader.device,
        loader.intent_map_rev,
        loader.slot_map_rev,
    )


GOLD = json.loads((FIXTURES / "two_commands.json").read_text())
FRESH = json.loads((FIXTURES / "two_commands_fresh.json").read_text())


@requires_torch
@pytest.mark.parametrize("text", GOLD["pairs"] + GOLD["shared"] + FRESH["pairs"])
def test_two_commands_are_found(read, text):
    assert multi.two_commands(text, read)


@requires_torch
@pytest.mark.parametrize("text", GOLD["singles"] + FRESH["singles"])
def test_one_command_is_not_split(read, text):
    assert multi.two_commands(text, read) is None


def test_every_word_of_the_checks_can_be_heard():
    grammar = set(
        (Path(__file__).parent.parent / "ML" / "vimaan_nlu" / "command_words.txt")
        .read_text()
        .split()
    )
    for fixture in (GOLD, FRESH):
        for kind in ("pairs", "shared", "singles"):
            for text in fixture.get(kind, []):
                assert set(text.split()) <= grammar, text
