import json
import itertools
import random

# -----------------------------
# 1. Seed commands per intent
# -----------------------------
seed_commands = {
    "GEAR_UP": ["gear up", "raise landing gear", "retract landing gear"],
    "GEAR_DOWN": ["gear down", "lower landing gear", "drop landing gear", "extend landing gear"],
    "FLAPS_UP": ["flaps up", "raise flaps"],
    "FLAPS_DOWN": ["flaps down", "lower flaps"],
    "AUTOPILOT_1_ON": ["autopilot 1 on", "engage autopilot 1"],
    "AUTOPILOT_1_OFF": ["autopilot 1 off", "disengage autopilot 1"],
    "AUTOPILOT_2_ON": ["autopilot 2 on", "engage autopilot 2"],
    "AUTOPILOT_2_OFF": ["autopilot 2 off", "disengage autopilot 2"],
    "FLIGHT_DIRECTOR_1_ON": ["flight director 1 on", "enable flight director 1"],
    "FLIGHT_DIRECTOR_1_OFF": ["flight director 1 off", "disable flight director 1"],
    "FLIGHT_DIRECTOR_2_ON": ["flight director 2 on", "enable flight director 2"],
    "FLIGHT_DIRECTOR_2_OFF": ["flight director 2 off", "disable flight director 2"],
    "PARKING_BRAKE_ON": ["parking brake on", "engage parking brake"],
    "PARKING_BRAKE_OFF": ["parking brake off", "release parking brake"],
    "ENGINE_1_ON": ["engine 1 on", "start engine 1"],
    "ENGINE_1_OFF": ["engine 1 off", "stop engine 1"],
    "ENGINE_2_ON": ["engine 2 on", "start engine 2"],
    "ENGINE_2_OFF": ["engine 2 off", "stop engine 2"]
}

# SCHEMA = {
#     "LANDING GEAR": {
#         "templates": [
#             "gear {state}",
#             "landing gear {state}"
#             "{state} gear",
#             "{state} landing gear",
#         ],
#         "slots": {
#             "state": {
#                 "type": "categorical",
#                 "values": ["up", "down", "raise", "drop", "extend", "retract"]
#             }
#         }
#     },
# }

# -----------------------------
# 2. Optional prefixes/suffixes
# -----------------------------
prefixes = ["Please", "Hey", "Could you", ""]
suffixes = ["now", "immediately", ""]

# -----------------------------
# 3. Templates
# -----------------------------
templates = [
    "{prefix} {command} {suffix}",
    "{command} {suffix}",
    "{prefix} {command}",
    "{command}"
]

# -----------------------------
# 4. Generate dataset
# -----------------------------
dataset = []

for intent, commands in seed_commands.items():
    for cmd in commands:
        for tpl in templates:
            for prefix in prefixes:
                for suffix in suffixes:
                    text = tpl.format(command=cmd, prefix=prefix, suffix=suffix)
                    text = " ".join(text.split())  # clean extra spaces
                    dataset.append({"text": text, "intent": intent, "slots": {}})

# Shuffle dataset
random.shuffle(dataset)

# -----------------------------
# 5. Write to JSONL
# -----------------------------
with open("aviation_cmds.jsonl", "w") as f:
    for entry in dataset:
        f.write(json.dumps(entry) + "\n")

print(f"Generated {len(dataset)} boolean command variations in boolean_commands_dataset.jsonl")
