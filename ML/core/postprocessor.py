import re

ACTION_STATE_MAP = {
    'raise': 'up',
    'lower': 'down',
    'retract': 'up',
    'extend': 'down',
    'stow': 'up',

    'engage': 'on',
    'disengage': 'off',
    'turn on': 'on',
    'turn off': 'off',
    'start': 'on',
    'stop': 'off',
    'shut': 'off',
    'activate': 'on',
    'deactivate': 'off',
}

IMPLICIT_STATE_INTENTS = {
    'toggle_landing_gear': True,
    'toggle_flaps': True,
    'toggle_autopilot_1': True,
    'toggle_autopilot_2': True,
    'toggle_autopilot_3': True,
    'toggle_engine_1': True,
    'toggle_engine_2': True,
    'toggle_engine_3': True,
    'toggle_engine_4': True,
    'toggle_parking_brake': True,
}

def extract_numbers_from_text(text):
    numbers = re.findall(r'\d+\.?\d*', text)
    return numbers

def extract_digit_sequence_frequency(text):
    digit_map = {
        'zero': '0', 'oh': '0',
        'one': '1', 'two': '2', 'three': '3',
        'four': '4', 'five': '5', 'six': '6',
        'seven': '7', 'eight': '8', 'niner': '9', 'nine': '9',
        'point': '.', 'decimal': '.'
    }
    
    if 'com' in text.lower() or 'frequency' in text.lower():
        words = text.lower().split()
        result = []
        found_sequence = False
        
        for i, word in enumerate(words):
            if word in digit_map:
                result.append(digit_map[word])
                found_sequence = True
            elif found_sequence and len(result) >= 3:
                freq_str = ''.join(result)
                try:
                    freq_val = float(freq_str)
                    if 118 <= freq_val <= 137:
                        return freq_str
                except:
                    pass
                result = []
                found_sequence = False
        
        if len(result) >= 3:
            freq_str = ''.join(result)
            try:
                freq_val = float(freq_str)
                if 118 <= freq_val <= 137:
                    return freq_str
            except:
                pass
    
    return None


def add_implicit_state(slots, original_text, intent):
    if intent not in IMPLICIT_STATE_INTENTS:
        return slots
    
    if 'state' in slots and slots['state']:
        return slots
    
    text_lower = original_text.lower()
    
    for action, state_value in sorted(ACTION_STATE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if action in text_lower:
            slots['state'] = state_value
            return slots
    
    return slots


def postprocess_slots(slots, original_text, intent=None):
    
    numbers = extract_numbers_from_text(original_text)
    
    for slot_name, slot_value in list(slots.items()):
        if slot_value is None or slot_value == '':
            continue
            
        slot_str = str(slot_value).strip()
        
        if ' ' in slot_str and slot_str.replace(' ', '').isdigit():
            cleaned = slot_str.replace(' ', '')
            slots[slot_name] = cleaned
            continue
        
        if slot_name == 'frequency':
            digit_freq = extract_digit_sequence_frequency(original_text)
            if digit_freq:
                slots[slot_name] = digit_freq
                continue
            
            freq_numbers = re.findall(r'\d+\.?\d*', original_text)
            for num in freq_numbers:
                if '.' in num:
                    val = float(num)
                    if 118 <= val <= 137: 
                        slots[slot_name] = num
                        break

        
        if slot_name in ['altitude', 'degrees', 'flight_level', 'com_port']:
            if len(numbers) > 0:
                if slot_name == 'altitude':
                    for num in numbers:
                        try:
                            val = int(float(num))
                            if 1000 <= val <= 50000:
                                slots[slot_name] = num
                                break
                        except:
                            pass
                            
                elif slot_name == 'degrees':
                    for num in numbers:
                        try:
                            val = int(float(num))
                            if 0 <= val <= 360:
                                slots[slot_name] = num
                                break
                        except:
                            pass
                            
                elif slot_name == 'flight_level':
                    for num in reversed(numbers): 
                        try:
                            val = int(float(num))
                            if 10 <= val <= 430:
                                slots[slot_name] = num
                                break
                        except:
                            pass

                elif slot_name == 'com_port':
                    for num in numbers:
                        try:
                            val = int(float(num))
                            if 1 <= val <= 4 and len(num) == 1: 
                                slots[slot_name] = num
                                break
                        except:
                            pass

    
    if intent:
        slots = add_implicit_state(slots, original_text, intent)
    
    return slots
