import os
import sys

import torch

ml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".")
sys.path.insert(0, ml_path)

from vimaan_nlu.inference import predict
from vimaan_nlu.model_loader import ModelLoader


def test_commands():
    test_commands = [
        # Easy
        ("set heading 270", "set_autopilot_heading"),
        ("climb to 15000 feet", "set_autopilot_altitude"),
        ("maintain flight level 210", "set_flight_level"),
        ("gear up", "toggle_landing_gear"),
        ("flaps down", "toggle_flaps"),
        ("autopilot 1 on", "toggle_autopilot_1"),
        ("engine 1 off", "toggle_engine_1"),
        ("parking brake on", "toggle_parking_brake"),
        # Medium
        ("fly heading 090", "set_autopilot_heading"),
        ("change altitude to 8000", "set_autopilot_altitude"),
        ("request flight level 350", "set_flight_level"),
        ("turn to 180 degrees", "set_autopilot_heading"),
        ("raise the landing gear", "toggle_landing_gear"),
        ("lower the flaps", "toggle_flaps"),
        ("engage autopilot 2", "toggle_autopilot_2"),
        ("set com 1 frequency 118.75", "set_com_frequency"),
        ("please climb to 12000 feet", "set_autopilot_altitude"),
        ("could you set heading 315", "set_autopilot_heading"),
        # Hard
        ("fly heading zero niner zero", "set_autopilot_heading"),
        ("set altitude twenty thousand", "set_autopilot_altitude"),
        ("tune com 1 one two three point four five", "set_com_frequency"),
        ("climb to flight level two hundred fifty", "set_flight_level"),
        ("set heading one hundred eighty degrees", "set_autopilot_heading"),
        ("descend to seven thousand five hundred feet", "set_autopilot_altitude"),
        # Edge cases
        ("uh heading to 360", "set_autopilot_heading"),
        ("can you set altitude 5000 feet", "set_autopilot_altitude"),
        ("please engage autopilot 1 now", "toggle_autopilot_1"),
        ("i think we should climb to 10000", "set_autopilot_altitude"),
        ("maybe turn right to 270 degrees", "set_autopilot_heading"),
        ("let's set heading 045 degrees", "set_autopilot_heading"),
        # Out of scope (no chit-chat intents are trained today; model should
        # surface low confidence or its best-guess intent — we just print).
        ("what is the weather", None),
        ("tell me something interesting", None),
        ("are we there yet", None),
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

    for text, expected_intent in test_commands:
        result = predict(
            text, loader.model, loader.tokenizer, device, loader.intent_map_rev, loader.slot_map_rev
        )

        actual_intent = result["intent"]
        confidence = result["confidence"]
        slots = result["slots"]

        if expected_intent is None:
            status = "SKIP"
            skipped += 1
            expected_label = "<out-of-scope>"
        elif actual_intent == expected_intent:
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

    total_scored = passed + failed
    print(f"\n\nResults: {passed}/{total_scored} passed, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    test_commands()
