import os
import json
import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

from core import JointIntentAndSlotModel
from utils import get_latest_model_path


class ModelLoader:
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.intent_map = None
        self.slot_map = None
        self.intent_map_rev = None
        self.slot_map_rev = None
    
    def load_maps(self, model_path):
        with open(f"{model_path}/intent_map.json", "r") as f:
            self.intent_map = json.load(f)
        
        with open(f"{model_path}/slot_map.json", "r") as f:
            self.slot_map = json.load(f)
        
        self.intent_map_rev = {v: k for k, v in self.intent_map.items()}
        self.slot_map_rev = {v: k for k, v in self.slot_map.items()}
        
        return {
            'intents': len(self.intent_map),
            'slots': len(self.slot_map)
        }
    
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        dims = self.load_maps(model_path)
        
        self.model = JointIntentAndSlotModel(
            num_intents=dims['intents'],
            num_slots=dims['slots']
        )
        
        self.model.bert_for_slots = DistilBertForTokenClassification.from_pretrained(model_path)
        
        intent_classifier_path = os.path.join(model_path, "intent_classifier.bin")
        if os.path.exists(intent_classifier_path):
            intent_classifier_state = torch.load(intent_classifier_path, map_location=self.device)
            self.model.intent_classifier.load_state_dict(intent_classifier_state)
        else:
            raise FileNotFoundError(f"Intent classifier not found at {intent_classifier_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        return {
            'model_path': model_path,
            'device': str(self.device),
            'intents_loaded': dims['intents'],
            'slots_loaded': dims['slots']
        }
    
    def load_tokenizer(self, model_path):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        return {'tokenizer_loaded': True}
    
    def load_all(self, model_path=None):
        if model_path is None:
            model_path = get_latest_model_path()
        
        if not model_path:
            raise FileNotFoundError("No model path provided and no latest model found")
        
        results = {
            'model': self.load_model(model_path),
            'tokenizer': self.load_tokenizer(model_path),
            'maps': {
                'intents': len(self.intent_map),
                'slots': len(self.slot_map)
            }
        }
        
        return results