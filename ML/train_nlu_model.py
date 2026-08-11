import json
import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DistilBertTokenizerFast
from utils import find_latest_version_path, get_model_versions_dir
from utils.slot_alignment import align_offsets_to_labels, find_slot_char_spans
from vimaan_nlu import (
    JointIntentAndSlotModel,
    build_manifest,
    normalize_aviation_input,
    normalize_dataset,
    write_manifest,
)


# DATASET PREPARATION
class AviationCommandDataset(Dataset):
    def __init__(self, data, tokenizer, intent_map, slot_map, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.intent_map = intent_map
        self.slot_map = slot_map
        self.max_length = max_length
        self.slot_map_rev = {v: k for k, v in self.slot_map.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Normalize the text the SAME way inference does (T1.4) so training sees
        # e.g. "ap 1 on" (as the model will at run time), not "ap one on".
        text = normalize_aviation_input(item["text"])
        intent_label = self.intent_map[item["intent"]]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"][0]
        offsets = encoding["offset_mapping"][0].tolist()

        # Offset-based BIO alignment (T1.1/T1.2): anchor each slot value at a
        # word boundary and label tokens by char-span containment — no more
        # split()/word_ids() mismatch or "on"-inside-"one" mislabeling.
        spans = find_slot_char_spans(text, item.get("slots"))
        slot_labels = np.array(align_offsets_to_labels(offsets, spans, self.slot_map), dtype=int)

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"][0],
            "intent_label": torch.tensor(intent_label, dtype=torch.long),
            "slot_labels": torch.tensor(slot_labels, dtype=torch.long),
        }


# THE TRAINING PROCESS
def train_model(
    dataset_path,
    *,
    num_epochs=10,
    lr=5e-5,
    batch_size=16,
    base_model="distilbert-base-uncased",
    max_length=64,
    patience=2,
    model_save_dir=None,
):
    # Reproducibility (T1.10): the pipeline was fully unseeded except the split.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading final dataset...")
    with open(dataset_path) as f:
        data = [json.loads(line) for line in f]

    data = normalize_dataset(data)
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)

    intents = sorted(list(set(item["intent"] for item in data)))
    intent_map = {name: i for i, name in enumerate(intents)}

    slots = set(["O"])
    for item in data:
        for slot_name in item["slots"]:
            slots.add(f"B-{slot_name}")
            slots.add(f"I-{slot_name}")
    slot_map = {name: i for i, name in enumerate(sorted(list(slots)))}

    tokenizer = DistilBertTokenizerFast.from_pretrained(base_model)
    train_dataset = AviationCommandDataset(
        train_data, tokenizer, intent_map, slot_map, max_length=max_length
    )
    val_dataset = AviationCommandDataset(
        val_data, tokenizer, intent_map, slot_map, max_length=max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointIntentAndSlotModel(num_intents=len(intent_map), num_slots=len(slot_map)).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    models_dir = model_save_dir or get_model_versions_dir()
    latest_version = 0
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            if item.startswith("v"):
                try:
                    ver = int(item[1:])
                    latest_version = max(latest_version, ver)
                except:
                    pass

    next_version = f"v{latest_version + 1}"
    model_save_path = os.path.join(models_dir, next_version)

    print("\nStarting training...")
    numOfEpochs = num_epochs
    for epoch in range(numOfEpochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_label"].to(device)
            slot_labels = batch["slot_labels"].to(device)

            loss, _, _ = model(input_ids, attention_mask, intent_labels, slot_labels)

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Validation]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                intent_labels = batch["intent_label"].to(device)
                slot_labels = batch["slot_labels"].to(device)

                loss, _, _ = model(input_ids, attention_mask, intent_labels, slot_labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Average Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            print(f"Validation loss improved! Saving best model to {model_save_path}")
            os.makedirs(model_save_path, exist_ok=True)
            tokenizer.save_pretrained(model_save_path)
            model.bert_for_slots.save_pretrained(model_save_path)
            torch.save(
                model.intent_classifier.state_dict(), f"{model_save_path}/intent_classifier.bin"
            )
            with open(f"{model_save_path}/intent_map.json", "w") as f:
                json.dump(intent_map, f)
            with open(f"{model_save_path}/slot_map.json", "w") as f:
                json.dump(slot_map, f)

            # Provenance sidecar (Phase 3 / B-015 manifest). Written every time
            # the "best" model is overwritten so it always reflects the actually
            # checkpointed weights.
            manifest = build_manifest(
                dataset_path=dataset_path,
                dataset_rows=data,
                hyperparams={
                    "max_length": max_length,
                    "epochs": numOfEpochs,
                    "lr": lr,
                    "batch_size": batch_size,
                    "optimizer": "AdamW",
                    "early_stopping_patience": patience,
                    "train_val_split": 0.85,
                    "random_state": 42,
                    "base_model": base_model,
                    "best_val_loss": float(best_val_loss),
                    "best_epoch": epoch + 1,
                },
            )
            write_manifest(model_save_path, manifest)
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Count: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the joint NLU model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a .jsonl training set. Defaults to the "
        "latest under ML/datasets/05_final_merged/.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--model-save-dir",
        type=str,
        default=None,
        help="Override the parent dir for the new v<N> checkpoint.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(script_dir, "datasets", "05_final_merged")
    BASE_FILENAME = os.path.join(DATA_DIR, "aviation_cmds_final_training_set.jsonl")

    if args.dataset:
        chosen = args.dataset
        if not os.path.exists(chosen):
            print(f"Error: dataset not found: {chosen}")
            raise SystemExit(2)
    else:
        print(f"Searching for the latest training dataset in '{DATA_DIR}'...")
        chosen = find_latest_version_path(BASE_FILENAME)
        if not (chosen and os.path.exists(chosen)):
            print(f"Error: Dataset not found in '{DATA_DIR}'.")
            print("Please ensure your merged dataset exists and the path is correct.")
            raise SystemExit(2)

    print(f"Found dataset: {os.path.basename(chosen)}")
    train_model(
        chosen,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        base_model=args.base_model,
        max_length=args.max_length,
        patience=args.patience,
        model_save_dir=args.model_save_dir,
    )
