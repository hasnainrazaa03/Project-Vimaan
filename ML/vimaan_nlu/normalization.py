import re

try:
    from word2number import w2n
except ImportError as _w2n_err:
    raise ImportError(
        "The 'word2number' package is required for vimaan_nlu.normalization. "
        "Install it from the project requirements: pip install -r requirements.txt"
    ) from _w2n_err


PHONETIC_MAP = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "niner": "9",
    "nine": "9",
    "point": ".",
    "decimal": ".",
}


# Single spoken digits (phonetic alphabet). Used to build the digit-run and
# group-form patterns below.
_DIGIT_WORDS = "zero|oh|one|two|three|four|five|six|seven|eight|niner|nine"

# Cardinal-number words for the compound pass. A run that contains a MAGNITUDE
# word is handed to word2number as a whole ("seven thousand five hundred"->7500);
# a run without one (a pure spoken-digit sequence like "two seven zero") is left
# for the phonetic pass to concatenate.
_CARDINAL_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "niner",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "and",
}
_MAGNITUDE_WORDS = {"hundred", "thousand", "million", "billion"}

# ATC group-form: two-or-more spoken digits then a magnitude, e.g.
# "one five thousand" = 15000, "two five hundred" = 2500. Must run BEFORE the
# phonetic pass, which would otherwise collapse "one five" -> "15" and leave a
# dangling "thousand" (the old code produced "15 1000").
_GROUP_FORM_RE = re.compile(
    rf"\b((?:{_DIGIT_WORDS})(?:[\s-]+(?:{_DIGIT_WORDS}))+)\s+(thousand|hundred)\b",
    flags=re.IGNORECASE,
)

_DECIMAL_SEQUENCE_RE = re.compile(
    rf"\b(?:(?:{_DIGIT_WORDS})[\s-]*)+(?:\s+(?:point|decimal)\s+)(?:(?:{_DIGIT_WORDS})[\s-]*)+\b",
    flags=re.IGNORECASE,
)
_PHONETIC_SEQUENCE_RE = re.compile(
    rf"\b((?:{_DIGIT_WORDS})(?:\s+(?:{_DIGIT_WORDS}))+)\b",
    flags=re.IGNORECASE,
)
_SIMPLE_NUMBER_RE = re.compile(
    r"\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven"
    r"|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
    r"|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
    r"(?:\s+(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|niner|ten|eleven"
    r"|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
    r"|thirty|forty|fifty|sixty|seventy|eighty|ninety))?\b",
    flags=re.IGNORECASE,
)


def _digits_only(words):
    return "".join(PHONETIC_MAP[w] for w in words if PHONETIC_MAP.get(w, ".") != ".")


def _group_form_repl(match):
    digits = _digits_only(re.split(r"[\s-]+", match.group(1).lower()))
    if not digits:
        return match.group(0)
    multiplier = 1000 if match.group(2).lower() == "thousand" else 100
    return str(int(digits) * multiplier)


def _convert_decimal_sequence(match):
    parts = re.split(r"\b(?:point|decimal)\b", match.group(0))
    result_parts = [_digits_only(p.split()) for p in parts]
    result_parts = [p for p in result_parts if p]
    if len(result_parts) == 2:
        return result_parts[0] + "." + result_parts[1]
    if len(result_parts) == 1:
        return result_parts[0]
    return match.group(0)


def _convert_phonetic_sequence(match):
    return _digits_only(match.group(0).split())


def _convert_word_number(match):
    try:
        return str(w2n.word_to_num(match.group(0)))
    except (ValueError, IndexError):
        return match.group(0)


def _convert_cardinals(text):
    """Convert maximal cardinal-word runs that include a magnitude word.

    Pure spoken-digit runs (no hundred/thousand) are left untouched for the
    phonetic pass, so "two seven zero" stays a digit run (270), while
    "seven thousand five hundred" becomes 7500 in one piece.
    """
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        if words[i].lower() in _CARDINAL_WORDS and words[i].lower() != "and":
            j = i
            while j < len(words) and words[j].lower() in _CARDINAL_WORDS:
                j += 1
            run = [w.lower() for w in words[i:j]]
            while run and run[-1] == "and":
                run = run[:-1]
                j -= 1
            if any(w in _MAGNITUDE_WORDS for w in run):
                try:
                    out.append(str(w2n.word_to_num(" ".join(run))))
                    i = j
                    continue
                except (ValueError, IndexError):
                    pass
        out.append(words[i])
        i += 1
    return " ".join(out)


def normalize_aviation_input(text):
    text_lower = text.lower()
    # 1. Decimals (frequencies): "one one eight decimal seven five" -> "118.75".
    #    Must precede the phonetic pass, which would collapse the integer part.
    text_lower = _DECIMAL_SEQUENCE_RE.sub(_convert_decimal_sequence, text_lower)
    # 2. ATC group-form altitudes: "one five thousand" -> "15000". Before the
    #    phonetic pass so "one five" is not collapsed to "15" first.
    text_lower = _GROUP_FORM_RE.sub(_group_form_repl, text_lower)
    # 3. Cardinals with a magnitude word: "seven thousand five hundred" -> "7500".
    text_lower = _convert_cardinals(text_lower)
    # 4. Phonetic digit runs: "two seven zero" -> "270".
    text_lower = _PHONETIC_SEQUENCE_RE.sub(_convert_phonetic_sequence, text_lower)
    # 5. Remaining single/paired word-numbers: "twenty" -> "20".
    text_lower = _SIMPLE_NUMBER_RE.sub(_convert_word_number, text_lower)
    return text_lower


def normalize_slot_value(value_str):
    value_str = str(value_str).lower().strip()

    if value_str.replace(".", "").replace("-", "").isdigit():
        if "." not in value_str:
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
        normalized = re.sub(r"\b" + word + r"\b", digit, normalized, flags=re.IGNORECASE)

    if normalized.replace(".", "").replace("-", "").isdigit():
        return normalized

    return value_str


def normalize_dataset_item(item):
    if "slots" in item:
        for slot_name, slot_value in item["slots"].items():
            item["slots"][slot_name] = normalize_slot_value(slot_value)
    return item


def normalize_dataset(data):
    return [normalize_dataset_item(item) for item in data]
