from collections import defaultdict


def validate_example(example, valid_intents, valid_slots):
    errors = []
    
    if "text" not in example:
        errors.append("Missing 'text'")
    if "intent" not in example:
        errors.append("Missing 'intent'")
    if "slots" not in example:
        errors.append("Missing 'slots'")
    
    if "intent" in example and example["intent"] not in valid_intents:
        errors.append(f"Invalid intent: {example['intent']}")
    
    if "slots" in example:
        for slot_name in example["slots"].keys():
            if slot_name not in valid_slots:
                errors.append(f"Unknown slot: {slot_name}")
    
    if "text" in example:
        if not example["text"] or not example["text"].strip():
            errors.append("Empty text")
    
    return len(errors) == 0, errors


def validate_dataset(dataset, valid_intents, valid_slots):
    
    report = {
        "total_examples": len(dataset),
        "valid_examples": 0,
        "invalid_examples": 0,
        "invalid_details": [],
        "intent_distribution": defaultdict(int),
        "slot_distribution": defaultdict(int),
        "text_stats": {
            "min_length": float('inf'),
            "max_length": 0,
            "total_length": 0,
        },
        "duplicate_texts": 0,
        "is_valid": True
    }
    
    seen_texts = set()
    
    for idx, example in enumerate(dataset):
        is_valid, errors = validate_example(example, valid_intents, valid_slots)
        
        if is_valid:
            report["valid_examples"] += 1
            
            intent = example.get("intent")
            report["intent_distribution"][intent] += 1
            
            for slot_name in example.get("slots", {}).keys():
                report["slot_distribution"][slot_name] += 1
            
            text = example.get("text", "")
            text_len = len(text)
            report["text_stats"]["min_length"] = min(report["text_stats"]["min_length"], text_len)
            report["text_stats"]["max_length"] = max(report["text_stats"]["max_length"], text_len)
            report["text_stats"]["total_length"] += text_len
            
            if text in seen_texts:
                report["duplicate_texts"] += 1
            seen_texts.add(text)
        
        else:
            report["invalid_examples"] += 1
            report["invalid_details"].append({
                "index": idx,
                "errors": errors
            })
    
    if report["valid_examples"] > 0:
        report["text_stats"]["avg_length"] = report["text_stats"]["total_length"] / report["valid_examples"]
    
    report["is_valid"] = report["invalid_examples"] == 0
    
    return report


def print_validation_report(report):
    
    status = "PASS" if report["is_valid"] else "FAIL"
    print(f"\n{status} Validation Summary")
    print("-" * 70)
    print(f"Total examples:     {report['total_examples']}")
    print(f"Valid examples:     {report['valid_examples']}")
    print(f"Invalid examples:   {report['invalid_examples']}")
    
    if report["invalid_examples"] > 0:
        print(f"\nFirst 3 invalid examples:")
        for invalid in report["invalid_details"][:3]:
            print(f"  Index {invalid['index']}: {invalid['errors']}")
    
    print(f"\nIntent Distribution:")
    for intent, count in sorted(report["intent_distribution"].items()):
        pct = (count / report["valid_examples"] * 100) if report["valid_examples"] > 0 else 0
        print(f"  {intent}: {count} ({pct:.1f}%)")
    
    print(f"\nText Length Stats:")
    print(f"  Min: {report['text_stats']['min_length']}")
    print(f"  Max: {report['text_stats']['max_length']}")
    if 'avg_length' in report['text_stats']:
        print(f"  Avg: {report['text_stats']['avg_length']:.1f}")
    
    if report["duplicate_texts"] > 0:
        print(f"\nFound {report['duplicate_texts']} duplicate texts")
    
    print()
