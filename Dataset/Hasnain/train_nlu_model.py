import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import os
from utils import find_latest_version_path, get_next_version_path

try:
    from num2words import num2words
except ImportError:
    print("Installing 'num2words' library...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "num2words"])
    from num2words import num2words

#DATASET PREPARATION
class AviationCommandDataset(Dataset):
    def __init__(self, data, tokenizer, intent_map, slot_map):
        self.data = data
        self.tokenizer = tokenizer
        self.intent_map = intent_map
        self.slot_map = slot_map
        self.slot_map_rev = {v: k for k, v in self.slot_map.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        intent_label = self.intent_map[item['intent']]
        
        encoding = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=64, return_tensors='pt'
        )

        slot_labels = np.ones(len(encoding['input_ids'][0]), dtype=int) * -100
        
        words = text.lower().split()
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        word_slot_map = {}
        for slot_name, slot_value in item['slots'].items():
            value_str = str(slot_value).lower()
            words_for_slot = []
            if value_str.isdigit():
                words_for_slot.append(value_str)
                words_for_slot.append(num2words(value_str))
            else:
                words_for_slot.extend(value_str.split())

            for i, word in enumerate(words_for_slot):
                bio_prefix = "B-" if i == 0 else "I-"
                word_slot_map[word] = self.slot_map[f"{bio_prefix}{slot_name}"]
        
        current_word_idx = 0
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            if not token.startswith("##"):
                if current_word_idx < len(words):
                    word = words[current_word_idx]
                    slot_labels[i] = word_slot_map.get(word, self.slot_map['O'])
                    current_word_idx += 1
            else:
                prev_label_id = slot_labels[i-1]
                if prev_label_id != -100 and self.slot_map_rev[prev_label_id] != 'O':
                    slot_name = self.slot_map_rev[prev_label_id].split('-')[1]
                    slot_labels[i] = self.slot_map.get(f"I-{slot_name}", self.slot_map['O'])
                else:
                    slot_labels[i] = self.slot_map['O']

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'intent_label': torch.tensor(intent_label, dtype=torch.long),
            'slot_labels': torch.tensor(slot_labels, dtype=torch.long)
        }

#THE MODEL ARCHITECTURE
class JointIntentAndSlotModel(torch.nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        self.bert_for_slots = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=num_slots)
        
        self.intent_classifier = torch.nn.Linear(self.bert_for_slots.config.hidden_size, num_intents)

    def forward(self, input_ids, attention_mask, intent_labels=None, slot_labels=None):
        bert_output = self.bert_for_slots.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        
        cls_token_output = sequence_output[:, 0, :]
        intent_logits = self.intent_classifier(cls_token_output)
  
        slot_logits = self.bert_for_slots.classifier(sequence_output)
        
        total_loss = 0
        if intent_labels is not None and slot_labels is not None:
            intent_loss = torch.nn.CrossEntropyLoss()(intent_logits, intent_labels.view(-1))
            slot_loss = torch.nn.CrossEntropyLoss()(slot_logits.view(-1, self.bert_for_slots.num_labels), slot_labels.view(-1))
            total_loss = intent_loss + slot_loss

        return total_loss, intent_logits, slot_logits

#THE TRAINING PROCESS
def train_model(dataset_path):
    print("Loading final dataset...")
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]
        
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)
    
    intents = sorted(list(set(item['intent'] for item in data)))
    intent_map = {name: i for i, name in enumerate(intents)}
    
    slots = set(['O'])
    for item in data:
        for slot_name in item['slots']:
            slots.add(f"B-{slot_name}")
            slots.add(f"I-{slot_name}") 
    slot_map = {name: i for i, name in enumerate(sorted(list(slots)))}

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = AviationCommandDataset(train_data, tokenizer, intent_map, slot_map)
    val_dataset = AviationCommandDataset(val_data, tokenizer, intent_map, slot_map)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointIntentAndSlotModel(num_intents=len(intent_map), num_slots=len(slot_map)).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 2
    
    base_model_save_path = os.path.join(os.path.dirname(dataset_path), "..", "..", "vimaan_nlu_model_best")
    model_save_path = get_next_version_path(base_model_save_path)

    print("\nStarting training...")
    numOfEpochs = 10
    for epoch in range(numOfEpochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_labels = batch['intent_label'].to(device)
            slot_labels = batch['slot_labels'].to(device)
            
            loss, _, _ = model(input_ids, attention_mask, intent_labels, slot_labels)
            
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent_labels = batch['intent_label'].to(device)
                slot_labels = batch['slot_labels'].to(device)
                
                loss, _, _ = model(input_ids, attention_mask, intent_labels, slot_labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            print(f"Validation loss improved! Saving best model to {model_save_path}")
            os.makedirs(model_save_path, exist_ok=True)
            tokenizer.save_pretrained(model_save_path)
            model.bert_for_slots.save_pretrained(model_save_path)
            torch.save(model.intent_classifier.state_dict(), f"{model_save_path}/intent_classifier.bin")
            with open(f"{model_save_path}/intent_map.json", "w") as f: json.dump(intent_map, f)
            with open(f"{model_save_path}/slot_map.json", "w") as f: json.dump(slot_map, f)
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Count: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break 

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(script_dir, "datasets", "05_final_merged")
    BASE_FILENAME = os.path.join(DATA_DIR, "aviation_cmds_final_training_set.jsonl")
    
    print(f"Searching for the latest training dataset in '{DATA_DIR}'...")
    latest_dataset = find_latest_version_path(BASE_FILENAME)
    
    if latest_dataset and os.path.exists(latest_dataset):
        print(f"Found dataset: {os.path.basename(latest_dataset)}")
        train_model(latest_dataset)
    else:
        print(f"Error: Dataset not found in '{DATA_DIR}'.")
        print("Please ensure your merged dataset exists and the path is correct.")
