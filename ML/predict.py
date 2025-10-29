import json
import torch
import os
import sys

ml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".")
sys.path.insert(0, ml_path)

from core.model_loader import ModelLoader
from core.inference import predict


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model...")
    loader = ModelLoader(device)
    results = loader.load_all()
    
    print(f"Model loaded: {results['model']}")
    print(f"Intents: {results['maps']['intents']}, Slots: {results['maps']['slots']}")
    print("Type 'quit' to exit.\n")
    
    while True:
        command = input("Enter command: ")
        if command.lower() == 'quit':
            break
        
        result = predict(
            command,
            loader.model,
            loader.tokenizer,
            device,
            loader.intent_map_rev,
            loader.slot_map_rev
        )
        
        print("\n--- Prediction ---")
        print(f"Intent: {result['intent']}")
        print(f"Slots: {json.dumps(result['slots'], indent=2)}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-------------------\n")
