import os
import json
import random
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

#MODEL SETUP

def setup_model():
    """Loads the Pegasus model and tokenizer from Hugging Face."""
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    print("Setting up the paraphrasing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = 'tuner007/pegasus_paraphrase'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    
    print("Model setup complete.")
    return tokenizer, model, device

#PARAPHRASING LOGIC

def paraphrase_command(text, tokenizer, model, device, num_variations=3, num_beams=5):

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
    INPUT_FILENAME = os.path.join(script_dir, "aviation_cmds.jsonl")
    OUTPUT_FILENAME = os.path.join(script_dir, "aviation_cmds_augmented.jsonl")
    
    tokenizer, model, device = setup_model()
    
    original_dataset = []
    with open(INPUT_FILENAME, "r") as f:
        for line in f:
            original_dataset.append(json.loads(line))
            
    augmented_dataset = []
    print(f"\nAugmenting {len(original_dataset)} original examples...")
    
    for entry in tqdm(original_dataset):
        augmented_dataset.append(entry)
        
        try:
            new_texts = paraphrase_command(entry['text'], tokenizer, model, device)
            
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