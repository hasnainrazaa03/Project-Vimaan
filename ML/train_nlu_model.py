import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DistilBertTokenizerFast
from utils import find_latest_version_path, get_model_versions_dir
from utils.device import resolve_device
from utils.run_logger import RunLogger
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
    device=None,
):
    # Reproducibility (T1.10): the pipeline was fully unseeded except the split.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading final dataset...")
    with open(dataset_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    data = normalize_dataset(data)

    # 3-way split: 70% train / 15% val (early stopping) / 15% held-out TEST.
    # The test split is never trained or early-stopped on and is saved with the
    # model (holdout_test.jsonl) so the evaluator can measure real generalization
    # instead of re-slicing the training file. Stratified by intent when possible.
    def _split(rows, frac, seed=42):
        labels = [r["intent"] for r in rows]
        try:
            return train_test_split(rows, test_size=frac, random_state=seed, stratify=labels)
        except ValueError:  # a class too small to stratify
            return train_test_split(rows, test_size=frac, random_state=seed)

    train_data, temp_data = _split(data, 0.30)
    val_data, test_data = _split(temp_data, 0.50)
    print(f"Split: train {len(train_data)}, val {len(val_data)}, held-out test {len(test_data)}")

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

    device = resolve_device(device)  # cuda > mps (Apple GPU) > cpu, or forced
    print(f"Using device: {device}")
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

    # Live metrics stream for the training monitor (ML/training_monitor.py).
    ml_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(ml_dir, "training_runs", next_version)
    logger = RunLogger(run_dir)
    logger.log(
        type="meta",
        version=next_version,
        device=str(device),
        dataset=os.path.basename(dataset_path),
        max_length=max_length,
        batch_size=batch_size,
        epochs=num_epochs,
        lr=lr,
        patience=patience,
        num_intents=len(intent_map),
        num_slots=len(slot_map),
        steps_per_epoch=len(train_loader),
        split={"train": len(train_data), "val": len(val_data), "test": len(test_data)},
        intent_dist=dict(Counter(r["intent"] for r in train_data).most_common()),
    )
    print(f"Live metrics -> {run_dir}/metrics.jsonl  (watch: python ML/training_monitor.py)")

    print("\nStarting training...")
    numOfEpochs = num_epochs
    start_time = time.time()
    global_step = 0
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

            step_loss = loss.item()
            total_train_loss += step_loss
            loss.backward()
            optimizer.step()

            global_step += 1
            logger.log(
                type="step",
                epoch=epoch + 1,
                step=global_step,
                loss=step_loss,
                elapsed=time.time() - start_time,
            )

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        intent_correct = 0
        intent_total = 0
        slot_true = []
        slot_pred = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Validation]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                intent_labels = batch["intent_label"].to(device)
                slot_labels = batch["slot_labels"].to(device)

                loss, intent_logits, slot_logits = model(
                    input_ids, attention_mask, intent_labels, slot_labels
                )
                total_val_loss += loss.item()

                intent_correct += (intent_logits.argmax(dim=-1) == intent_labels).sum().item()
                intent_total += intent_labels.numel()

                mask = slot_labels != -100
                slot_true.extend(slot_labels[mask].cpu().tolist())
                slot_pred.extend(slot_logits.argmax(dim=-1)[mask].cpu().tolist())

        avg_val_loss = total_val_loss / len(val_loader)
        val_intent_acc = intent_correct / max(intent_total, 1)
        val_slot_f1 = (
            f1_score(slot_true, slot_pred, average="macro", zero_division=0) if slot_true else 0.0
        )
        print(
            f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}  "
            f"Intent Acc: {val_intent_acc:.4f}  Slot F1(macro): {val_slot_f1:.4f}"
        )
        is_best = avg_val_loss < best_val_loss
        logger.log(
            type="epoch",
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_intent_acc=val_intent_acc,
            val_slot_f1=float(val_slot_f1),
            best=bool(is_best),
            elapsed=time.time() - start_time,
        )

        if is_best:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            print(f"Validation loss improved! Saving best model to {model_save_path}")
            os.makedirs(model_save_path, exist_ok=True)
            tokenizer.save_pretrained(model_save_path)
            model.bert_for_slots.save_pretrained(model_save_path)
            # .cpu() so the checkpoint is device-agnostic (e.g. saved from MPS).
            torch.save(
                {k: v.detach().cpu() for k, v in model.intent_classifier.state_dict().items()},
                f"{model_save_path}/intent_classifier.bin",
            )
            with open(f"{model_save_path}/intent_map.json", "w") as f:
                json.dump(intent_map, f)
            with open(f"{model_save_path}/slot_map.json", "w") as f:
                json.dump(slot_map, f)

            # Held-out test set (never trained or early-stopped on) so the
            # evaluator can measure true generalization — ML/evaluation/evaluator
            # loads this in preference to re-slicing the training file.
            with open(f"{model_save_path}/holdout_test.jsonl", "w", encoding="utf-8") as f:
                for row in test_data:
                    f.write(json.dumps(row) + "\n")

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

    logger.log(
        type="done",
        best_val_loss=float(best_val_loss),
        elapsed=time.time() - start_time,
        model_path=model_save_path,
    )
    logger.close()
    print(f"\nTraining complete. Best val loss {best_val_loss:.4f} -> {model_save_path}")


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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a device: cpu | mps | cuda. Default: auto (cuda > mps > cpu).",
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
        device=args.device,
    )
