import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import os
from core import normalize_aviation_input, JointIntentAndSlotModel, postprocess_slots
from utils import find_latest_version_path, get_latest_model_path


try:
    from word2number import w2n
except ImportError:
    print("Installing word2number...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "word2number"])
    from word2number import w2n

def reconstruct_slot_value(tokens):
    
    if not tokens:
        return ""
    
    reconstructed = ""
    
    for i, token in enumerate(tokens):
        clean_token = token.replace("##", "")
        
        if clean_token == ".":
            reconstructed += clean_token
        elif token.startswith("##"):
            reconstructed += clean_token
        else:
            if i == 0:
                reconstructed = clean_token
            else:
                reconstructed += " " + clean_token
    
    return reconstructed.strip()

def predict(text, model, tokenizer, device, intent_map_rev, slot_map_rev):
    
    text = normalize_aviation_input(text)

    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        _, intent_logits, slot_logits = model(input_ids, attention_mask)


    intent_pred_idx = torch.argmax(intent_logits, dim=1).item()
    intent_pred = intent_map_rev[intent_pred_idx]
    
    slot_pred_indices = torch.argmax(slot_logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    extracted_slots = {}
    current_slot_name = None
    current_slot_tokens = [] 
    
    for token, slot_idx in zip(tokens, slot_pred_indices):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        slot_name_bio = slot_map_rev[slot_idx]
        
        if slot_name_bio.startswith("B-"):
            if current_slot_name and current_slot_tokens:
                extracted_slots[current_slot_name] = reconstruct_slot_value(current_slot_tokens)
            
            current_slot_name = slot_name_bio[2:]
            current_slot_tokens = [token]
        
        elif slot_name_bio.startswith("I-") and current_slot_name:
            slot_name_from_tag = slot_name_bio[2:]
            if slot_name_from_tag == current_slot_name:
                current_slot_tokens.append(token) 
        
        else:
            if current_slot_name and current_slot_tokens:
                extracted_slots[current_slot_name] = reconstruct_slot_value(current_slot_tokens)
                current_slot_name = None
                current_slot_tokens = []
    
    if current_slot_name and current_slot_tokens:
        extracted_slots[current_slot_name] = reconstruct_slot_value(current_slot_tokens)
    
    extracted_slots = postprocess_slots(extracted_slots, text, intent_pred)

    return intent_pred, extracted_slots


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    print(f"Searching for the latest model...")
    model_path = get_latest_model_path()
    
    if not model_path or not os.path.exists(model_path):
        print(f"Error: No trained model found.")
    else:
        print(f"Loading model from: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with open(f"{model_path}/intent_map.json", "r") as f:
            intent_map = json.load(f)
        with open(f"{model_path}/slot_map.json", "r") as f:
            slot_map = json.load(f)

        intent_map_rev = {v: k for k, v in intent_map.items()}
        slot_map_rev = {v: k for k, v in slot_map.items()}

        model = JointIntentAndSlotModel(
            num_intents=len(intent_map), 
            num_slots=len(slot_map)
        )
        
        model.bert_for_slots = DistilBertForTokenClassification.from_pretrained(model_path)
        
        model.intent_classifier.load_state_dict(
            torch.load(f"{model_path}/intent_classifier.bin", map_location=device)
        )
        
        model.to(device)
        model.eval()

        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        
        print("Model loaded successfully!")
        print("Type 'quit' to exit.\n")

        while True:
            command = input("Enter command: ")
            if command.lower() == 'quit':
                break
                
            pred_intent, pred_slots = predict(
                command, model, tokenizer, device, intent_map_rev, slot_map_rev
            )
            
            print("\n--- Prediction ---")
            print(f"Intent: {pred_intent}")
            print(f"Slots: {json.dumps(pred_slots, indent=2)}")
            print("--------------------\n")
