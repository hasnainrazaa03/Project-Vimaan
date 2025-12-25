import random
import re


def generate_dynamic_value(tag):
    if tag == "<DYNAMIC>":  # COM Radio
        base = random.randint(118, 136)
        decimal = random.choice([0, 25, 50, 75, 5, 10, 15, 20, 30, 40, 60, 80, 90])
        return f"{base}.{decimal:02d}"
    
    elif tag == "<DYNAMIC_NAV>":  # NAV Radio
        base = random.randint(108, 117)
        decimal = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 80, 90])
        return f"{base}.{decimal:02d}"
    
    elif tag == "<DYNAMIC_ADF>":  # ADF
        return str(random.randint(190, 1750))
    
    elif tag == "<DYNAMIC_SQUAWK>":  # Transponder
        return "".join([str(random.randint(0, 7)) for _ in range(4)])
    
    elif tag == "<DYNAMIC_BARO>":  # Barometer
        if random.random() > 0.5:
            val = 28.0 + (random.random() * 3.0)
            return f"{val:.2f}"
        else:
            return str(random.randint(950, 1050))
    
    return "0"


def fill_template(template, intent_data, specific_slot_values=None):
    filled_template = template
    slots_data = {}
    
    placeholders = re.findall(r'\{(\w+)\}', template)
    
    for placeholder in placeholders:
        mapping = intent_data.get("placeholder_mapping", {}).get(placeholder, {})
        word_to_speak = None
        
        if specific_slot_values and placeholder in specific_slot_values:
            slot_value = specific_slot_values[placeholder]
            
            if placeholder == "value":
                slots_data["value"] = slot_value
                word_to_speak = slot_value
            
            elif placeholder == "position":
                slots_data["position"] = slot_value
                position_mapping = mapping.get(slot_value, {})
                synonyms = position_mapping.get("synonyms", [slot_value])
                word_to_speak = random.choice(synonyms)
            
            elif placeholder == "state":
                slots_data["state"] = slot_value
                state_mapping = mapping.get(slot_value, {})
                synonyms = state_mapping.get("synonyms", [slot_value])
                word_to_speak = random.choice(synonyms)
            
            elif placeholder == "action":
                action_mapping = mapping.get(slot_value, {})
                if isinstance(action_mapping, dict):
                    synonyms = action_mapping.get("synonyms", [slot_value])
                else:
                    synonyms = [slot_value]
                word_to_speak = random.choice(synonyms)
                if isinstance(action_mapping, dict):
                    state_from_action = action_mapping.get("state", slot_value)
                    if "state" in intent_data.get("slots", {}):
                        slots_data["state"] = state_from_action
            
            else:
                word_to_speak = slot_value
        
        else:
            
            if placeholder == "value":
                slot_info = intent_data.get("slots", {}).get(placeholder, {})
                if slot_info.get("type") == "numeric":
                    slot_value = str(random.randint(slot_info.get("min", 0), slot_info.get("max", 100)))
                    slots_data["value"] = slot_value
                    word_to_speak = slot_value
            
            elif placeholder == "unit":
                if isinstance(mapping, dict) and "synonyms" in mapping:
                    synonyms = mapping.get("synonyms", [])
                    word_to_speak = random.choice(synonyms) if synonyms else placeholder
                else:
                    word_to_speak = placeholder
            
            elif placeholder == "control":
                if isinstance(mapping, dict) and "synonyms" in mapping:
                    synonyms = mapping.get("synonyms", [])
                    word_to_speak = random.choice(synonyms) if synonyms else placeholder
                else:
                    word_to_speak = placeholder
            
            elif placeholder == "action":
                
                if isinstance(mapping, dict):
                    sample_key = list(mapping.keys())[0] if mapping else None
                    
                    if sample_key and isinstance(mapping[sample_key], dict):
                        action_type = random.choice(list(mapping.keys())) 
                        action_mapping = mapping[action_type]
                        
                        synonyms = action_mapping.get("synonyms", [action_type])
                        word_to_speak = random.choice(synonyms)
                        
                        slot_value = action_mapping.get("state", action_type)
                        if "state" in intent_data.get("slots", {}):
                            slots_data["state"] = slot_value
                    else:
                        synonyms = mapping.get("synonyms", [])
                        word_to_speak = random.choice(synonyms) if synonyms else "action"
                else:
                    word_to_speak = "action"
            
            elif placeholder == "state":
                state_value = random.choice(intent_data.get("slots", {}).get("state", {}).get("values", ["selected"]))
                state_mapping = mapping.get(state_value, {})
                synonyms = state_mapping.get("synonyms", [state_value])
                word_to_speak = random.choice(synonyms)
                slots_data["state"] = state_value
            
            elif placeholder == "position":
                position_value = random.choice(intent_data.get("slots", {}).get("position", {}).get("values", ["0"]))
                position_mapping = mapping.get(position_value, {})
                synonyms = position_mapping.get("synonyms", [position_value])
                word_to_speak = random.choice(synonyms)
                slots_data["position"] = position_value
            
            else:
                if isinstance(mapping, dict) and "synonyms" in mapping:
                    synonyms = mapping.get("synonyms", [])
                    word_to_speak = random.choice(synonyms) if synonyms else placeholder
                else:
                    word_to_speak = placeholder
        
        if word_to_speak:
            filled_template = filled_template.replace("{" + placeholder + "}", word_to_speak)
    
    filled_template = " ".join(filled_template.split()).strip()
    
    return filled_template, slots_data
