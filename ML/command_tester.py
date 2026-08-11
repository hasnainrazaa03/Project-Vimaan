import os
import sys

import torch

ml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".")
sys.path.insert(0, ml_path)

from vimaan_nlu.inference import predict
from vimaan_nlu.model_loader import ModelLoader


def test_commands():
    # (text, expected_intent, expected_slots). expected_slots must be a SUBSET
    # of the predicted slots — this now catches slot regressions (e.g. a spoken
    # altitude decoded to the wrong number), which the intent-only check missed.
    test_commands = [
        # Easy
        ("set heading 270", "set_autopilot_heading", {"degrees": "270"}),
        ("climb to 15000 feet", "set_autopilot_altitude", {"altitude": "15000"}),
        ("maintain flight level 210", "set_flight_level", {"flight_level": "210"}),
        ("gear up", "toggle_landing_gear", {"state": "up"}),
        ("flaps down", "toggle_flaps", {"state": "down"}),
        ("autopilot 1 on", "toggle_autopilot_1", {"state": "on"}),
        ("engine 1 off", "toggle_engine_1", {"state": "off"}),
        ("parking brake on", "toggle_parking_brake", {"state": "on"}),
        # Medium
        ("fly heading 090", "set_autopilot_heading", {"degrees": "090"}),
        ("change altitude to 8000", "set_autopilot_altitude", {"altitude": "8000"}),
        ("request flight level 350", "set_flight_level", {"flight_level": "350"}),
        ("turn to 180 degrees", "set_autopilot_heading", {"degrees": "180"}),
        ("raise the landing gear", "toggle_landing_gear", {"state": "up"}),
        ("lower the flaps", "toggle_flaps", {"state": "down"}),
        ("engage autopilot 2", "toggle_autopilot_2", {"state": "on"}),
        ("set com 1 frequency 118.75", "set_com_frequency", {"frequency": "118.75"}),
        ("please climb to 12000 feet", "set_autopilot_altitude", {"altitude": "12000"}),
        ("could you set heading 315", "set_autopilot_heading", {"degrees": "315"}),
        # Hard
        ("fly heading zero niner zero", "set_autopilot_heading", {"degrees": "090"}),
        ("set altitude twenty thousand", "set_autopilot_altitude", {"altitude": "20000"}),
        ("tune com 1 one two three point four five", "set_com_frequency", {"frequency": "123.45"}),
        ("climb to flight level two hundred fifty", "set_flight_level", {"flight_level": "250"}),
        ("set heading one hundred eighty degrees", "set_autopilot_heading", {"degrees": "180"}),
        (
            "descend to seven thousand five hundred feet",
            "set_autopilot_altitude",
            {"altitude": "7500"},
        ),
        # Edge cases
        ("uh heading to 360", "set_autopilot_heading", {"degrees": "360"}),
        ("can you set altitude 5000 feet", "set_autopilot_altitude", {"altitude": "5000"}),
        ("please engage autopilot 1 now", "toggle_autopilot_1", {"state": "on"}),
        ("i think we should climb to 10000", "set_autopilot_altitude", {"altitude": "10000"}),
        ("maybe turn right to 270 degrees", "set_autopilot_heading", {"degrees": "270"}),
        ("let's set heading 045 degrees", "set_autopilot_heading", {"degrees": "045"}),
        # Out of scope (no chit-chat intents are trained today; model should
        # surface low confidence or its best-guess intent — we just print).
        ("what is the weather", None, None),
        ("tell me something interesting", None, None),
        ("are we there yet", None, None),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    loader = ModelLoader(device)
    results = loader.load_all()
    print(f"Model loaded from: {results['model']['model_path']}")
    print(f"Intents: {results['maps']['intents']}, Slots: {results['maps']['slots']}\n")
    print("Model loaded!\n")

    passed = 0
    failed = 0
    skipped = 0

    for text, expected_intent, expected_slots in test_commands:
        result = predict(
            text, loader.model, loader.tokenizer, device, loader.intent_map_rev, loader.slot_map_rev
        )

        actual_intent = result["intent"]
        confidence = result["confidence"]
        slots = result["slots"]

        slot_miss = None
        if expected_slots:
            slot_miss = {
                k: (v, slots.get(k)) for k, v in expected_slots.items() if slots.get(k) != v
            }

        if expected_intent is None:
            status = "SKIP"
            skipped += 1
            expected_label = "<out-of-scope>"
        elif actual_intent == expected_intent and not slot_miss:
            status = "PASS"
            passed += 1
            expected_label = expected_intent
        else:
            status = "FAIL"
            failed += 1
            expected_label = expected_intent

        print(
            f"{status} | {text:40s} | Expected: {expected_label:30s} | Got: {actual_intent:30s} | Conf: {confidence:.2f}"
        )
        if slots:
            print(f"       Slots: {slots}")
        if slot_miss:
            print(f"       SLOT MISMATCH (expected: got): {slot_miss}")

    total_scored = passed + failed
    print(f"\n\nResults: {passed}/{total_scored} passed, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    test_commands()
