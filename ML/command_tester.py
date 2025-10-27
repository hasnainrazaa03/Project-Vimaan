import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import os
from core import normalize_aviation_input, JointIntentAndSlotModel, postprocess_slots
from utils import find_latest_version_path, get_latest_model_path
from predict import predict

print(f"Searching for the latest model...")
model_path = get_latest_model_path()

if not model_path or not os.path.exists(model_path):
    print(f"Error: No trained model found.")
    print("Please train a model first using 'train_nlu_model.py'.")
else:
    print(f"Loading model from: {model_path}\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(f"{model_path}/intent_map.json", "r") as f:
        intent_map = json.load(f)
    with open(f"{model_path}/slot_map.json", "r") as f:
        slot_map = json.load(f)

    intent_map_rev = {v: k for k, v in intent_map.items()}
    slot_map_rev = {v: k for k, v in slot_map.items()}

    model = JointIntentAndSlotModel(
        num_intents=len(intent_map), num_slots=len(slot_map)
    )
    model.intent_classifier.load_state_dict(
        torch.load(f"{model_path}/intent_classifier.bin", map_location=device)
    )
    model.bert_for_slots = DistilBertForTokenClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    print("Model loaded successfully!\n")

    # Test commands
    test_commands = [
        # Easy
        "set heading 270",
        "climb to 15000 feet",
        "maintain flight level 210",
        "gear up",
        "flaps down",
        "autopilot 1 on",
        "engine 1 off",
        "parking brake on",

        # Medium
        "fly heading 090",
        "change altitude to 8000",
        "request flight level 350",
        "turn to 180 degrees",
        "raise the landing gear",
        "lower the flaps",
        "engage autopilot 2",
        "set com 1 frequency 118.75",
        "please climb to 12000 feet",
        "could you set heading 315",

        # Hard
        "fly heading zero niner zero",
        "set altitude twenty thousand",
        "tune com 1 one two three point four five",
        "climb to flight level two hundred fifty",
        "set heading one hundred eighty degrees",
        "descend to seven thousand five hundred feet",

        # Edge cases
        "uh heading to 360",
        "can you set altitude 5000 feet",
        "please engage autopilot 1 now",
        "i think we should climb to 10000",
        "maybe turn right to 270 degrees",
        "let's set heading 045 degrees",

        # Out of scope
        "what is the weather",
        "how are you doing",
        "what time is it",
        "tell me something interesting",
        "are we there yet",
    ]

    for cmd in test_commands:
        print(f"\n{'='*60}")
        print(f"Testing: {cmd}")
        print('='*60)
        pred_intent, pred_slots = predict(cmd, model, tokenizer, device, intent_map_rev, slot_map_rev)
        print(f"Intent: {pred_intent}")
        print(f"Slots: {json.dumps(pred_slots, indent=2)}")