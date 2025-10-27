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

COMPOUND_NUMBER_PATTERNS = [
    (r'\b((?:zero|one|two|three|four|five|six|seven|eight|niner|nine)\s+hundred(?:\s+(?:and\s+)?(?:zero|one|two|three|four|five|six|seven|eight|niner|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety))?)\b', 'compound_hundred'),
    (r'\b((?:zero|one|two|three|four|five|six|seven|eight|niner|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:\s+(?:hundred|thousand|million))+(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|niner|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand))*)\b', 'compound_large'),
]


NUMBER_WORD_PATTERN = r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b'


def normalize_aviation_input(text):

    text_lower = text.lower()
    
    def convert_phrase(match):
        phrase = match.group(1)
        try:
            result = w2n.word_to_num(phrase)
            return str(result)
        except:
            return phrase
    
    try:
        text_lower = re.sub(
            r'\b((?:zero|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:\s+(?:hundred|thousand|million|billion))*(?:\s+(?:and\s+)?(?:zero|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety))?)\b',
            convert_phrase,
            text_lower,
            flags=re.IGNORECASE
        )
    except:
        pass
    
    digit_sequence_pattern = r'\b((?:zero|oh|one|two|three|four|five|six|seven|eight|niner|nine)\s+){2,}(?:zero|oh|one|two|three|four|five|six|seven|eight|niner|nine)\b'

    def convert_digit_sequence(match):
        phrase = match.group(0)
        digits = []
        for word in phrase.split():
            if word in PHONETIC_MAP:
                digit = PHONETIC_MAP[word]
                if digit != '.': 
                    digits.append(digit)
        return ''.join(digits)

    text_lower = re.sub(digit_sequence_pattern, convert_digit_sequence, text_lower, flags=re.IGNORECASE)

    for word, digit in PHONETIC_MAP.items():
        text_lower = re.sub(r'\b' + word + r'\b', digit, text_lower, flags=re.IGNORECASE)

    
    def convert_word_number(match):
        phrase = match.group(0)
        try:
            result = w2n.word_to_num(phrase)
            return str(result)
        except:
            return phrase
    
    try:
        text_lower = re.sub(
            r'\b((?:zero|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:\s+hundred)?(?:\s+(?:and\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety))?(?:\s+thousand)(?:\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety))?(?:\s+hundred)?(?:\s+(?:and\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety))?|(?:zero|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:\s+hundred)(?:\s+(?:and\s+)?(?:zero|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety))?)\b',
            convert_word_number,
            text_lower,
            flags=re.IGNORECASE
        )
    except:
        pass
    
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
        normalized = normalized.replace(word, digit)
    
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
