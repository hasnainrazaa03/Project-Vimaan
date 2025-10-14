import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import os
from utils import find_latest_version_path

class JointIntentAndSlotModel(torch.nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        from transformers import DistilBertForTokenClassification
        self.bert_for_slots = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=num_slots)
        self.intent_classifier = torch.nn.Linear(self.bert_for_slots.config.hidden_size, num_intents)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_for_slots.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        cls_token_output = sequence_output[:, 0, :]
        intent_logits = self.intent_classifier(cls_token_output)
        slot_logits = self.bert_for_slots.classifier(sequence_output)
        return intent_logits, slot_logits

#Main Prediction Logic
def predict(text, model, tokenizer, device, intent_map_rev, slot_map_rev):
    """Takes a text command and returns the predicted intent and slots."""
    
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
        intent_logits, slot_logits = model(input_ids, attention_mask)

    intent_pred_idx = torch.argmax(intent_logits, dim=1).item()
    intent_pred = intent_map_rev[intent_pred_idx]
    
    slot_pred_indices = torch.argmax(slot_logits, dim=2)[0].cpu().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    extracted_slots = {}
    current_slot_name = None
    current_slot_value = ""

    for token, slot_idx in zip(tokens, slot_pred_indices):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        slot_name_bio = slot_map_rev[slot_idx]
        
        if slot_name_bio.startswith("B-"):
            if current_slot_name:
                extracted_slots[current_slot_name] = current_slot_value.strip()
            
            current_slot_name = slot_name_bio[2:]
            current_slot_value = token.replace('##', '')
        
        elif slot_name_bio.startswith("I-") and current_slot_name:
            current_slot_value += token.replace('##', '')
        
        else: 
            if current_slot_name:
                extracted_slots[current_slot_name] = current_slot_value.strip()
                current_slot_name = None
                current_slot_value = ""

    if current_slot_name:
        extracted_slots[current_slot_name] = current_slot_value.strip()

    return intent_pred, extracted_slots

if __name__ == "__main__":
    
    script_dir = os.path.dirname(__file__)
    BASE_MODEL_PATH = os.path.join(script_dir, "vimaan_nlu_model")
    print(f"Searching for the latest model...")
    model_path = find_latest_version_path(BASE_MODEL_PATH)
    if not model_path or not os.path.exists(model_path):
        print(f"Error: No trained model folder found matching the pattern '{BASE_MODEL_PATH}_vX'.")
        print("Please train a model first using 'train_nlu_model.py'.")
    else:
        print(f"Loading model from: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        print("Loading model...")
        with open(f"{model_path}/intent_map.json", "r") as f:
            intent_map = json.load(f)
        with open(f"{model_path}/slot_map.json", "r") as f:
            slot_map = json.load(f)

        intent_map_rev = {v: k for k, v in intent_map.items()}
        slot_map_rev = {v: k for k, v in slot_map.items()}

        model = JointIntentAndSlotModel(num_intents=len(intent_map), num_slots=len(slot_map))
        model.intent_classifier.load_state_dict(torch.load(f"{model_path}/intent_classifier.bin"))
        model.bert_for_slots = DistilBertForTokenClassification.from_pretrained(model_path)
        model.to(device)
        model.eval() 

        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        print("Model loaded successfully. You can now enter commands.")
        print("Type 'quit' to exit.")

        while True:
            command = input("\nEnter command: ")
            if command.lower() == 'quit':
                break
                
            pred_intent, pred_slots = predict(command, model, tokenizer, device, intent_map_rev, slot_map_rev)
            
            print("\n--- Prediction ---")
            print(f"Intent: {pred_intent}")
            print(f"Slots: {json.dumps(pred_slots, indent=2)}")
            print("--------------------")