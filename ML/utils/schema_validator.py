
def validate_schema(schema):
    errors = []
    
    if not schema or not isinstance(schema, dict):
        return False, ["Schema must be a non-empty dictionary"]
    
    for intent_key, intent_data in schema.items():
        required_fields = ["intent", "type", "templates", "slots", "placeholder_mapping"]
        for field in required_fields:
            if field not in intent_data:
                errors.append(f"{intent_key}: Missing '{field}' field")
        
        if "templates" in intent_data:
            if not isinstance(intent_data["templates"], list) or len(intent_data["templates"]) == 0:
                errors.append(f"{intent_key}: 'templates' must be non-empty list")
        
        if "slots" in intent_data:
            if not isinstance(intent_data["slots"], dict):
                errors.append(f"{intent_key}: 'slots' must be dictionary")
            
            for slot_name, slot_def in intent_data["slots"].items():
                if "type" not in slot_def:
                    errors.append(f"{intent_key}/slots/{slot_name}: Missing 'type'")
                if "values" not in slot_def and slot_def.get("type") != "numeric":
                    errors.append(f"{intent_key}/slots/{slot_name}: Missing 'values' for non-numeric slot")
    
    return len(errors) == 0, errors


def get_all_intents(schema):
    return [v["intent"] for v in schema.values()]


def get_all_slots(schema):
    slots = {}
    for intent_data in schema.values():
        for slot_name, slot_details in intent_data.get("slots", {}).items():
            if slot_name not in slots:
                slots[slot_name] = slot_details
    return slots


def print_schema_info(schema):
    print("\n" + "="*70)
    print("SCHEMA INFORMATION")
    print("="*70)
    print(f"Total intents: {len(schema)}")
    print(f"Total unique slots: {len(get_all_slots(schema))}")
    print(f"\nIntents:")
    for key, data in schema.items():
        intent_type = data.get("type", "unknown")
        print(f"  - {data['intent']} ({intent_type})")
    print()
