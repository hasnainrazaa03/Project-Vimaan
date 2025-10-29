import torch
import numpy as np
from core import normalize_aviation_input, postprocess_slots


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


def extract_slots(slot_pred_indices, tokens, slot_map_rev):
    extracted_slots = {}
    current_slot_name = None
    current_slot_tokens = []
    
    for token, slot_idx in zip(tokens, slot_pred_indices):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        
        slot_name_bio = slot_map_rev.get(int(slot_idx), 'O')
        
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
    
    return extracted_slots


def predict(text, model, tokenizer, device, intent_map_rev, slot_map_rev, do_postprocess=True):

    text_normalized = normalize_aviation_input(text)
    
    encoding = tokenizer(
        text_normalized,
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
    intent_confidence = torch.softmax(intent_logits, dim=1)[0, intent_pred_idx].item()
    
    slot_pred_indices = torch.argmax(slot_logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    extracted_slots = extract_slots(slot_pred_indices, tokens, slot_map_rev)
    
    if do_postprocess:
        extracted_slots = postprocess_slots(extracted_slots, text_normalized, intent_pred)
    
    return {
        'intent': intent_pred,
        'slots': extracted_slots,
        'confidence': intent_confidence,
        'original_text': text,
        'normalized_text': text_normalized
    }