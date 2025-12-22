import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm

# -------------------------
# CONFIGURATION
# -------------------------
DATA_DIR = Path(r"D:\\Project-Vimaan\\Dataset\\f_output")
FILES = [
    "base_cmds.jsonl",
    "flanT5_cmds.jsonl",
    "parrot_cmds.jsonl",
    "pegasus_cmds.jsonl",
]
OUTPUT_FILE = DATA_DIR / "aviation_cmds.jsonl"

# Filtering thresholds (tweak as needed)
MIN_WORDS = 2               # drop too-short entries
MAX_CHARS = 200             # drop too-long text entries
MAX_REPEAT_COUNT = 10       # drop if any single token repeats > this
SHUFFLE_SEED = 42           # set to None for non-deterministic shuffle

# Regexes
# Matches lines like: "altitude = 1000" or "heading=123.4" with optional whitespace around '='
ASSIGN_NUMBER_RE = re.compile(r'^[\s\w\-\+]+=\s*-?\d+(\.\d+)?\s*$', flags=re.IGNORECASE)

# matches the same word repeated many times, e.g., "altitude altitude altitude ..."
REPEATED_WORD_RE = re.compile(r'\b(\w+)\b', flags=re.IGNORECASE)

# Optional: catch obviously nonsense punctuation repetitions like "!!!!!!" or "altitude=altitude+..."
REPEATED_SUBSTRING_RE = re.compile(r'(\b\w+\b)(?:.*\1){%d,}' % (MAX_REPEAT_COUNT-1), flags=re.IGNORECASE)

# -------------------------
# HELPERS
# -------------------------
def normalize_text(s: str) -> str:
    return s.strip()

def looks_like_assignment_number(text: str) -> bool:
    # strip trailing punctuation, then test
    t = text.strip()
    return bool(ASSIGN_NUMBER_RE.match(t))

def too_many_repeats(text: str, max_repeat=MAX_REPEAT_COUNT) -> bool:
    # token frequency
    tokens = REPEATED_WORD_RE.findall(text.lower())
    if not tokens:
        return False
    freq = Counter(tokens)
    if freq.most_common(1)[0][1] > max_repeat:
        return True
    # fallback: repeated-substring regex (catches long spans of same token)
    if REPEATED_SUBSTRING_RE.search(text):
        return True
    return False

def is_valid_entry(entry: dict, drop_reasons: dict) -> bool:
    # basic structure
    if not isinstance(entry, dict):
        drop_reasons['not_dict'] += 1
        return False

    text = entry.get("text", "")
    intent = entry.get("intent")
    slots = entry.get("slots", {})

    if text is None:
        drop_reasons['no_text_field'] += 1
        return False

    text = normalize_text(text)
    if not text:
        drop_reasons['empty_text'] += 1
        return False

    if not intent:
        drop_reasons['no_intent'] += 1
        return False

    # length checks
    if len(text) > MAX_CHARS:
        drop_reasons['too_long'] += 1
        return False

    # word count
    word_count = len(text.split())
    if word_count < MIN_WORDS:
        drop_reasons['too_short'] += 1
        return False

    # assignment to a number like "altitude = 1000"
    if looks_like_assignment_number(text):
        drop_reasons['assign_number'] += 1
        return False

    # repeated token checks (e.g., altitude repeated 20 times)
    if too_many_repeats(text):
        drop_reasons['repeated_token'] += 1
        return False

    # slots must be dict (you already specified this earlier)
    if not isinstance(slots, dict):
        drop_reasons['bad_slots'] += 1
        return False

    # pass
    return True

# -------------------------
# LOAD, FILTER, DEDUP, SHUFFLE, SAVE
# -------------------------
combined = []
seen_texts = set()
global_drop_reasons = defaultdict(int)
per_file_stats = {}

for fname in FILES:
    path = DATA_DIR / fname
    file_drops = defaultdict(int)
    file_loaded = 0
    file_kept = 0
    if not path.exists():
        print(f"Warning: {path} not found, skipping.")
        per_file_stats[fname] = {"loaded": 0, "kept": 0, "dropped": 0}
        continue

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {fname}"):
            file_loaded += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                file_drops['json_error'] += 1
                continue

            # temporary per-entry drop reasons collector (to increment global counters cleanly)
            temp_reasons = defaultdict(int)
            if is_valid_entry(entry, temp_reasons):
                text_key = normalize_text(entry["text"]).lower()
                if text_key in seen_texts:
                    file_drops['duplicate_text'] += 1
                else:
                    seen_texts.add(text_key)
                    combined.append(entry)
                    file_kept += 1
            else:
                # add temp_reasons to both file and global counters
                for k, v in temp_reasons.items():
                    file_drops[k] += v

    total_dropped = file_loaded - file_kept
    per_file_stats[fname] = {"loaded": file_loaded, "kept": file_kept, "dropped": total_dropped, "drop_details": dict(file_drops)}
    for k, v in file_drops.items():
        global_drop_reasons[k] += v

# Shuffle
if SHUFFLE_SEED is not None:
    random.Random(SHUFFLE_SEED).shuffle(combined)
else:
    random.shuffle(combined)

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for entry in combined:
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Reporting summary
print("=== Summary ===")
print(f"Files processed: {len(FILES)}")
print(f"Total kept entries: {len(combined)}")
print(f"Output saved to: {OUTPUT_FILE}")
print("\nPer-file stats:")
for fname, s in per_file_stats.items():
    print(f"  {fname}: loaded={s['loaded']}, kept={s['kept']}, dropped={s['dropped']}")
    if s.get('drop_details'):
        for reason, count in s['drop_details'].items():
            print(f"    - {reason}: {count}")

print("\nGlobal drop reasons:")
for reason, count in sorted(global_drop_reasons.items(), key=lambda x: -x[1]):
    print(f"  {reason}: {count}")
