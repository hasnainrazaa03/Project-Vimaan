"""Gold end-to-end NLU regression (Phase 6E).

Runs canonical utterances through the REAL model (ModelLoader + predict) and
asserts intent + key slots. This is the inference-level regression net that the
pure-Python tests can't provide — it catches a model/normalization/postprocessor
/dispatch regression that unit tests miss.

Marked ``requires_torch`` (skipped in CI) and skips when no trained model is
present. Expectations were verified against production model v10.

The audio variant (``tests/test_gold_audio.py``) adds STT over recorded wavs.
"""

import pytest

requires_torch = pytest.mark.requires_torch

# (utterance, expected_intent, expected_slots-subset)
GOLD = [
    ("set heading two seven zero", "set_autopilot_heading", {"degrees": "270"}),
    ("fly heading zero niner zero", "set_autopilot_heading", {"degrees": "090"}),
    ("climb to flight level three five zero", "set_flight_level", {"flight_level": "350"}),
    ("set altitude twelve thousand", "set_autopilot_altitude", {}),
    ("gear up", "toggle_landing_gear", {"state": "up"}),
    ("lower the landing gear", "toggle_landing_gear", {"state": "down"}),
    ("flaps down", "set_flaps", {"position": "down"}),
    ("retract flaps", "set_flaps", {"position": "up"}),
    ("engage autopilot one", "toggle_autopilot", {"state": "on", "instance": "1"}),
    ("autopilot two off", "toggle_autopilot", {"state": "off", "instance": "2"}),
    ("flight director one on", "toggle_flight_director", {"state": "on", "instance": "1"}),
    ("parking brake on", "toggle_parking_brake", {"state": "on"}),
    ("start engine one", "toggle_engine", {"state": "on", "instance": "1"}),
    ("shut down engine two", "toggle_engine", {"state": "off", "instance": "2"}),
    (
        "tune com one to one one eight decimal seven five",
        "set_com_frequency",
        {"com_port": "1", "frequency": "118.75"},
    ),
    ("what time is it", "ask_time", {}),
    # Two-digit headings, and a stray number that must not replace the value.
    ("change heading to eighty", "set_autopilot_heading", {"degrees": "80"}),
    ("change heading to eighty for me", "set_autopilot_heading", {"degrees": "80"}),
    ("heading forty five", "set_autopilot_heading", {"degrees": "45"}),
    ("heading zero eight zero", "set_autopilot_heading", {"degrees": "080"}),
    ("eighty fly heading one eight four", "set_autopilot_heading", {"degrees": "184"}),
    ("altitude two thousand", "set_autopilot_altitude", {"altitude": "2000"}),
    ("gear up", "toggle_landing_gear", {"state": "up"}),
    ("tune com two to one two one decimal five", "set_com_frequency", {"com_port": "2"}),
    # E1: the port named after the frequency (v17 left it untagged; in flight
    # COM 1 was tuned).
    (
        "frequency one three three decimal one on com two",
        "set_com_frequency",
        {"com_port": "2", "frequency": "133.1"},
    ),
    # Tier 1 (v18): the Part C flight checklist's phrases. Kept in GOLD_TIER1.
]

GOLD_TIER1 = [
    ("speed one six zero", "set_autopilot_speed", {"speed": "160"}),
    ("mach point seven eight", "set_autopilot_speed", {"mach": "0.78"}),
    (
        "descend at one thousand five hundred feet per minute",
        "set_vertical_speed",
        {"vs": "1500", "direction": "down"},
    ),
    (
        "climb at eight hundred feet per minute",
        "set_vertical_speed",
        {"vs": "800", "direction": "up"},
    ),
    ("heading mode", "set_autopilot_mode", {"mode": "heading"}),
    ("approach mode", "set_autopilot_mode", {"mode": "approach"}),
    ("level change", "set_autopilot_mode", {"mode": "level change"}),
    ("altitude hold mode", "set_autopilot_mode", {"mode": "altitude hold"}),
    ("engage v nav", "set_autopilot_mode", {"mode": "v nav"}),
    ("auto throttle on", "toggle_autothrottle", {"state": "on"}),
    ("auto throttle off", "toggle_autothrottle", {"state": "off"}),
    ("altimeter two niner niner two", "set_altimeter", {"pressure": "2992"}),
    ("q n h one zero one three", "set_altimeter", {"pressure": "1013"}),
    ("altimeter standard", "set_altimeter", {"setting": "standard"}),
    ("flaps fifteen", "set_flaps", {"flap_setting": "15"}),
    ("flaps full", "set_flaps", {"position": "full"}),
]
# Tier 2 (v20): radios and transponder.
GOLD_TIER2 = [
    (
        "com two standby one two one decimal niner",
        "set_com_standby",
        {"com_port": "2", "frequency": "121.9"},
    ),
    ("flip com one", "swap_com_frequency", {"com_port": "1"}),
    (
        "tune nav one to one one zero decimal five",
        "set_nav_frequency",
        {"nav_port": "1", "nav_frequency": "110.5"},
    ),
    (
        "nav two standby one one three decimal niner",
        "set_nav_standby",
        {"nav_port": "2", "nav_frequency": "113.9"},
    ),
    ("swap nav two", "swap_nav_frequency", {"nav_port": "2"}),
    ("course zero niner zero", "set_course", {"degrees": "090"}),
    ("nav two course one eight zero", "set_course", {"degrees": "180", "nav_port": "2"}),
    ("squawk four two one seven", "set_transponder_code", {"squawk": "4217"}),
    ("squawk zero one two zero", "set_transponder_code", {"squawk": "0120"}),
    ("squawk twelve hundred", "set_transponder_code", {"squawk": "1200"}),
    ("transponder alt", "set_transponder_mode", {"xpdr_mode": "alt"}),
    ("transponder mode charlie", "set_transponder_mode", {"xpdr_mode": "alt"}),
    ("squawk standby", "set_transponder_mode", {"xpdr_mode": "standby"}),
    ("ident", "set_transponder_mode", {"xpdr_mode": "ident"}),
    ("squawk ident", "set_transponder_mode", {"xpdr_mode": "ident"}),
]
# Tier 3 (v26): lights, cabin signs, yaw damper, speed brake, autobrake.
GOLD_TIER3 = [
    ("landing lights on", "toggle_lights", {"light": "landing", "switch": "on"}),
    ("strobes on", "toggle_lights", {"light": "strobe", "switch": "on"}),
    ("beacon off", "toggle_lights", {"light": "beacon", "switch": "off"}),
    ("turn the taxi lights off", "toggle_lights", {"light": "taxi", "switch": "off"}),
    ("nav lights on", "toggle_lights", {"light": "nav", "switch": "on"}),
    ("all lights off", "toggle_lights", {"light": "all", "switch": "off"}),
    ("strobes auto", "toggle_lights", {"light": "strobe", "switch": "auto"}),
    ("seat belt sign on", "toggle_signs", {"sign": "seat belt", "switch": "on"}),
    ("seat belt sign auto", "toggle_signs", {"sign": "seat belt", "switch": "auto"}),
    ("no smoking sign off", "toggle_signs", {"sign": "no smoking", "switch": "off"}),
    ("yaw damper on", "toggle_yaw_damper", {"state": "on"}),
    ("speed brake armed", "set_speed_brake", {"speedbrake": "armed"}),
    ("stow the speed brakes", "set_speed_brake", {"speedbrake": "retract"}),
    ("speed brakes up", "set_speed_brake", {"speedbrake": "up"}),
    ("auto brake two", "set_autobrake", {"autobrake_level": "2"}),
    ("auto brake max", "set_autobrake", {"autobrake": "max"}),
    ("auto brake r t o", "set_autobrake", {"autobrake": "r t o"}),
    ("pitot heat on", "toggle_anti_ice", {"ice": "pitot", "switch": "on"}),
    ("probe heat on", "toggle_anti_ice", {"ice": "pitot", "switch": "on"}),
    ("engine anti ice on", "toggle_anti_ice", {"ice": "engine", "switch": "on"}),
    ("wing anti ice off", "toggle_anti_ice", {"ice": "wing", "switch": "off"}),
    ("boots on", "toggle_anti_ice", {"ice": "wing", "switch": "on"}),
    ("window heat on", "toggle_anti_ice", {"ice": "window", "switch": "on"}),
    ("prop heat off", "toggle_anti_ice", {"ice": "prop", "switch": "off"}),
    ("avionics master on", "toggle_electrical", {"power": "avionics", "switch": "on"}),
    ("battery on", "toggle_electrical", {"power": "battery", "switch": "on"}),
    (
        "generator two off",
        "toggle_electrical",
        {"power": "generator", "instance": "2", "switch": "off"},
    ),
    ("alternators on", "toggle_electrical", {"power": "generator", "switch": "on"}),
    ("fuel pumps on", "toggle_fuel_pump", {"switch": "on"}),
    ("fuel pump one off", "toggle_fuel_pump", {"instance": "1", "switch": "off"}),
    ("boost pumps on", "toggle_fuel_pump", {"switch": "on"}),
]
GOLD = GOLD + GOLD_TIER1 + GOLD_TIER2 + GOLD_TIER3

# Bare numbers and fragments must never be acted on (scripts/model_gate.py
# checks these through the plugin's decision rules).
MUST_NOT_ACT = [
    "eighty",
    "seventy",
    "forty five",
    "eighty two on",
    "two forty",
    "on",
    "down",
    "point seven eight",
    "auto",
    "armed",
]


@pytest.fixture(scope="session")
def gold_model():
    from utils import get_latest_model_path

    if not get_latest_model_path():
        pytest.skip("no trained model present")
    from vimaan_nlu.model_loader import ModelLoader

    loader = ModelLoader()
    loader.load_all()
    return loader


@requires_torch
@pytest.mark.parametrize("text,intent,slots", GOLD, ids=[c[0] for c in GOLD])
def test_gold_intent_and_slots(gold_model, text, intent, slots):
    from vimaan_nlu import predict

    known = set(gold_model.intent_map_rev.values())
    if (text, intent, slots) in GOLD_TIER1 and "set_autopilot_speed" not in known:
        pytest.skip("the installed model predates Tier 1")  # the gate still requires these
    if (text, intent, slots) in GOLD_TIER2 and "set_transponder_code" not in known:
        pytest.skip("the installed model predates Tier 2")
    if (text, intent, slots) in GOLD_TIER3 and "set_autobrake" not in known:
        pytest.skip("the installed model predates Tier 3")

    r = predict(
        text,
        gold_model.model,
        gold_model.tokenizer,
        gold_model.device,
        gold_model.intent_map_rev,
        gold_model.slot_map_rev,
    )
    assert r["intent"] == intent, f"{text!r}: got {r['intent']} (conf {r['confidence']:.2f})"
    for key, val in slots.items():
        got = r["slots"].get(key)
        assert got == val, f"{text!r}: slot {key}={got!r}, expected {val!r}"
