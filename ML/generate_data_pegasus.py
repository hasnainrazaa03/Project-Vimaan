import os
import json
import random
import torch
from tqdm import tqdm
import huggingface_hub
from utils import get_next_version_path, find_latest_version_path

#MODEL SETUP
def setup_model():
    """Loads the Pegasus model and tokenizer from Hugging Face."""
    print("Setting up the paraphrasing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = 'tuner007/pegasus_paraphrase'
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    
    print("Model setup complete.")
    return tokenizer, model, device

#PARAPHRASING LOGIC
def paraphrase_command(text, slots, tokenizer, model, device, num_variations=3, num_beams=5):
    """Generates paraphrases using the Pegasus model."""
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=128, truncation=True).to(device)

    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=num_beams,
        num_return_sequences=num_variations,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    paraphrased_texts = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    return paraphrased_texts

#MAIN SCRIPT
if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    INPUT_DIR = os.path.join(script_dir, "datasets", "01_base")
    OUTPUT_DIR = os.path.join(script_dir, "datasets", "02_augmented_pegasus")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    latest_input_file = find_latest_version_path(os.path.join(INPUT_DIR, "aviation_cmds.jsonl"))
    
    if latest_input_file:
        BASE_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "aviation_cmds_augmented_pegasus.jsonl")
        OUTPUT_FILENAME = get_next_version_path(BASE_OUTPUT_FILENAME)

        print(f"Processing input: {latest_input_file}")
        print(f"Saving output to: {OUTPUT_FILENAME}")
    
        tokenizer, model, device = setup_model()
    
        original_dataset = []
        with open(latest_input_file, "r") as f:
            for line in f:
                original_dataset.append(json.loads(line))
            
        augmented_dataset = []
        print(f"\nAugmenting {len(original_dataset)} original examples with Pegasus...")
        
        for entry in tqdm(original_dataset):
            augmented_dataset.append(entry)
            try:
                new_texts = paraphrase_command(entry['text'], entry['slots'], tokenizer, model, device)
                for new_text in new_texts:
                    new_entry = {
                        "text": new_text,
                        "intent": entry['intent'],
                        "slots": entry['slots']
                    }
                    augmented_dataset.append(new_entry)
            except Exception as e:
                print(f"Skipping an entry due to error: {e}")

        random.shuffle(augmented_dataset)
        
        with open(OUTPUT_FILENAME, "w") as f:
            for entry in augmented_dataset:
                f.write(json.dumps(entry) + "\n")
                
        print("\n--- Augmentation Complete! ---")
        print(f"Original examples: {len(original_dataset)}")
        print(f"Total augmented examples: {len(augmented_dataset)}")
        print(f"Dataset saved to '{OUTPUT_FILENAME}'")
    else:
        print(f"Error: No base dataset file found in '{INPUT_DIR}'. Please run 'generate_slot_dataset.py' first.")