import os
import json
import random
import torch
from tqdm import tqdm
import huggingface_hub
from utils import get_next_version_path, find_latest_version_path

#MODEL SETUP
def setup_model():
    """Loads the FLAN-T5 model and tokenizer from Hugging Face."""
    print("Setting up the FLAN-T5 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model_name = 'google/flan-t5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    
    print("Model setup complete.")
    return tokenizer, model, device

#GENERATIVE LOGIC
def generate_variations(intent, slots, tokenizer, model, device, num_variations=3, num_beams=5):
    """Generates command text variations from a given intent and slots using FLAN-T5."""
    slot_string = ", ".join([f"{k}: {v}" for k, v in slots.items()])
    prompt = f"Generate a short, natural-sounding pilot voice command for the action '{intent}' with these parameters: {slot_string}"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(
        input_ids,
        max_length=32,
        num_beams=num_beams,
        num_return_sequences=num_variations,
        early_stopping=True
    )
    generated_texts = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    return generated_texts

#MAIN SCRIPT
if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    INPUT_DIR = os.path.join(script_dir, "datasets", "01_base")
    OUTPUT_DIR = os.path.join(script_dir, "datasets", "03_augmented_flan_t5")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    latest_input_file = find_latest_version_path(os.path.join(INPUT_DIR, "aviation_cmds.jsonl"))
    
    if latest_input_file:
        BASE_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "aviation_cmds_augmented_flan_t5.jsonl")
        OUTPUT_FILENAME = get_next_version_path(BASE_OUTPUT_FILENAME)

        print(f"Processing input: {latest_input_file}")
        print(f"Saving output to: {OUTPUT_FILENAME}")
    
        tokenizer, model, device = setup_model()
    
        original_dataset = []
        with open(latest_input_file, "r") as f:
            for line in f:
                original_dataset.append(json.loads(line))
            
        augmented_dataset = []
        print(f"\nGenerating variations for {len(original_dataset)} original examples using FLAN-T5...")
        
        for entry in tqdm(original_dataset):
            augmented_dataset.append(entry)
            try:
                new_texts = generate_variations(entry['intent'], entry['slots'], tokenizer, model, device)
                for new_text in new_texts:
                    if all(str(v).lower() in new_text.lower() for v in entry['slots'].values()):
                        augmented_dataset.append({
                            "text": new_text,
                            "intent": entry['intent'],
                            "slots": entry['slots']
                        })
            except Exception as e:
                print(f"Skipping an entry due to error: {e}")

        random.shuffle(augmented_dataset)
        
        with open(OUTPUT_FILENAME, "w") as f:
            for entry in augmented_dataset:
                f.write(json.dumps(entry) + "\n")
                
        print("\n--- Generation Complete! ---")
        print(f"Original examples: {len(original_dataset)}")
        print(f"Total augmented examples: {len(augmented_dataset)}")
        print(f"Dataset saved to '{OUTPUT_FILENAME}'")
    else:
        print(f"Error: No base dataset file found in '{INPUT_DIR}'. Please run 'generate_slot_dataset.py' first.")