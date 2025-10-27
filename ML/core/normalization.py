import re

try:
    from word2number import w2n
except ImportError:
    print("Installing word2number...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "word2number"])
    from word2number import w2n


PHONETIC_MAP = {
    'zero': '0', 'oh': '0',
    'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6',
    'seven': '7', 'eight': '8',
    'niner': '9', 'nine': '9',
    'point': '.', 'decimal': '.'
}


def normalize_aviation_input(text):
    text_lower = text.lower()
    def convert_phonetic_sequence(match):
        """Convert phonetic digit sequence like 'zero niner zero' to '090'"""
        phrase = match.group(0)
        digits = []
        for word in phrase.split():
            if word in PHONETIC_MAP:
                digit = PHONETIC_MAP[word]
                if digit != '.':
                    digits.append(digit)
        return ''.join(digits)
    
    phonetic_sequence_pattern = r'\b((?:zero|oh|one|two|three|four|five|six|seven|eight|niner|nine)(?:\s+(?:zero|oh|one|two|three|four|five|six|seven|eight|niner|nine))+)\b'
    text_lower = re.sub(phonetic_sequence_pattern, convert_phonetic_sequence, text_lower, flags=re.IGNORECASE)
    
    def convert_digit_sequence_with_decimal(match):
        phrase = match.group(0)
        
        parts = re.split(r'\b(?:point|decimal)\b', phrase)
        
        result_parts = []
        for part_idx, part in enumerate(parts):
            digits = []
            for word in part.split():
                word = word.strip()
                if word in PHONETIC_MAP:
                    digit = PHONETIC_MAP[word]
                    if digit != '.':
                        digits.append(digit)
            
            if digits:
                result_parts.append(''.join(digits))
        
        if len(result_parts) == 2:
            return result_parts[0] + '.' + result_parts[1]
        elif len(result_parts) == 1:
            return result_parts[0]
        
        return phrase
    
    decimal_sequence_pattern = r'\b(?:(?:zero|oh|one|two|three|four|five|six|seven|eight|niner|nine)[\s-]*)+(?:\s+(?:point|decimal)\s+)(?:(?:zero|oh|one|two|three|four|five|six|seven|eight|niner|nine)[\s-]*)+\b'
    text_lower = re.sub(decimal_sequence_pattern, convert_digit_sequence_with_decimal, text_lower, flags=re.IGNORECASE)

    def convert_compound_numbers(match):
        phrase = match.group(0)
        try:
            result = w2n.word_to_num(phrase)
            return str(result)
        except:
            return phrase
    
    compound_pattern = r'\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:\s+(?:and\s+)?)?)+(?:\s+(?:hundred|thousand|million|billion))(?:\s+(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:\s+(?:and\s+)?)?)+(?:\s+(?:hundred))?)?(?:\s+(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:\s+(?:and\s+)?)?)+)?\b'
    text_lower = re.sub(compound_pattern, convert_compound_numbers, text_lower, flags=re.IGNORECASE)
    
    def convert_word_number(match):
        phrase = match.group(0)
        try:
            result = w2n.word_to_num(phrase)
            return str(result)
        except:
            return phrase
    
    simple_number_pattern = r'\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)(?:\s+(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety))?\b'
    text_lower = re.sub(simple_number_pattern, convert_word_number, text_lower, flags=re.IGNORECASE)
    
    return text_lower


def normalize_slot_value(value_str):
    value_str = str(value_str).lower().strip()
    
    if value_str.replace('.', '').replace('-', '').isdigit():
        if '.' not in value_str:
            try:
                return str(int(value_str))
            except:
                return value_str
        return value_str
    
    try:
        result = w2n.word_to_num(value_str)
        return str(result)
    except:
        pass
    
    normalized = value_str
    for word, digit in PHONETIC_MAP.items():
        normalized = re.sub(r'\b' + word + r'\b', digit, normalized, flags=re.IGNORECASE)
    
    if normalized.replace('.', '').replace('-', '').isdigit():
        return normalized
    
    return value_str


def normalize_dataset_item(item):
    if 'slots' in item:
        for slot_name, slot_value in item['slots'].items():
            item['slots'][slot_name] = normalize_slot_value(slot_value)
    return item


def normalize_dataset(data):
    return [normalize_dataset_item(item) for item in data]
