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
]

# Bare numbers and fragments must never be acted on (scripts/model_gate.py
# checks these through the plugin's decision rules).
MUST_NOT_ACT = ["eighty", "seventy", "forty five", "eighty two on", "two forty", "on", "down"]


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
