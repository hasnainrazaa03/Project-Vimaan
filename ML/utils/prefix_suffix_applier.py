import random


def create_prefix_suffix_variants(dataset, config, augmentation_rate=0.25):
    examples_by_intent = {}
    for example in dataset:
        intent = example.get("intent")
        if intent not in examples_by_intent:
            examples_by_intent[intent] = []
        examples_by_intent[intent].append(example)
    
    augmented_examples = []
    
    for intent, examples in examples_by_intent.items():
        num_to_augment = max(1, int(len(examples) * augmentation_rate))
        
        indices_to_augment = random.sample(range(len(examples)), num_to_augment)
        
        for idx in indices_to_augment:
            original_example = examples[idx]
            text = original_example.get("text", "")
            
            new_text = apply_prefix_suffix_to_example(text, intent, config)
            
            new_example = {
                "text": new_text,
                "intent": original_example.get("intent"),
                "slots": original_example.get("slots").copy()
            }
            
            augmented_examples.append(new_example)
    
    return augmented_examples


def apply_prefix_suffix_to_example(text, intent, config):
    
    intent_rules = config.get("intent_rules", {}).get(intent, {})
    
    if not intent_rules.get("applicable", False):
        return text
    
    general_rules = config["general"]
    rand = random.random()
    
    apply_prefix_only = general_rules.get("apply_prefix_only", 0.40)
    apply_suffix_only = general_rules.get("apply_suffix_only", 0.40)
    apply_both = general_rules.get("apply_both", 0.20)
    
    if rand < apply_prefix_only:
        prefixes = intent_rules.get("prefixes", [])
        if prefixes:
            prefix = random.choice(prefixes)
            text = f"{prefix} {text}"
    
    elif rand < apply_prefix_only + apply_suffix_only:
        suffixes = intent_rules.get("suffixes", [])
        if suffixes:
            suffix = random.choice(suffixes)
            text = f"{text} {suffix}"
    
    else:
        prefixes = intent_rules.get("prefixes", [])
        suffixes = intent_rules.get("suffixes", [])
        if prefixes:
            prefix = random.choice(prefixes)
            text = f"{prefix} {text}"
        if suffixes:
            suffix = random.choice(suffixes)
            text = f"{text} {suffix}"
    
    return text
