"""Learning-curve experiment: does MORE training data help v11's weak spot?

Trains the joint model on increasing fractions of the SAME train split (val and
test held fixed and identical to what train_nlu_model.py produces at seed 42),
then measures intent accuracy and slot PAIR-F1 on the fixed held-out test set.
If the curve is still climbing at 100%, more data would help; if it has
plateaued, the bottleneck is data *diversity/quality*, not quantity.

Run (MPS auto-selected)::

    python ML/experiments/learning_curve.py
    python ML/experiments/learning_curve.py --fractions 0.1,0.25,0.5,1.0 --eval-sample 3000

Writes ML/experiments/results/learning_curve.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from train_nlu_model import AviationCommandDataset  # noqa: E402
from utils import find_latest_version_path  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from vimaan_nlu import (  # noqa: E402
    JointIntentAndSlotModel,
    normalize_dataset,
    normalize_slot_value,
    predict,
)

TOK = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def split3(data, seed=42):
    """Reproduce train_nlu_model.py's 70/15/15 split so `test` == v11's holdout."""

    def _split(rows, frac):
        labels = [r["intent"] for r in rows]
        try:
            return train_test_split(rows, test_size=frac, random_state=seed, stratify=labels)
        except ValueError:
            return train_test_split(rows, test_size=frac, random_state=seed)

    train, temp = _split(data, 0.30)
    val, test = _split(temp, 0.50)
    return train, val, test


def subsample(rows, frac, seed=42):
    if frac >= 1.0:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    keep = set(idx[: int(len(rows) * frac)])
    return [r for i, r in enumerate(rows) if i in keep]


def train_quick(
    train_data, val_data, intent_map, slot_map, device, *, epochs, lr, bs, max_len, patience
):
    model = JointIntentAndSlotModel(num_intents=len(intent_map), num_slots=len(slot_map)).to(device)
    opt = AdamW(model.parameters(), lr=lr)
    tr = DataLoader(
        AviationCommandDataset(train_data, TOK, intent_map, slot_map, max_length=max_len),
        batch_size=bs,
        shuffle=True,
    )
    vl = DataLoader(
        AviationCommandDataset(val_data, TOK, intent_map, slot_map, max_length=max_len),
        batch_size=bs,
    )
    best, no_improve, best_state = 1e9, 0, None
    for ep in range(epochs):
        model.train()
        for b in tr:
            opt.zero_grad()
            loss, _, _ = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["intent_label"].to(device),
                b["slot_labels"].to(device),
            )
            loss.backward()
            opt.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for b in vl:
                loss, _, _ = model(
                    b["input_ids"].to(device),
                    b["attention_mask"].to(device),
                    b["intent_label"].to(device),
                    b["slot_labels"].to(device),
                )
                vloss += loss.item()
        vloss /= max(1, len(vl))
        if vloss < best:
            best, no_improve = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_test(model, test_data, intent_map, slot_map, device, *, sample, seed=42):
    rev_i = {v: k for k, v in intent_map.items()}
    rev_s = {v: k for k, v in slot_map.items()}
    rng = random.Random(seed)
    rows = test_data[:]
    rng.shuffle(rows)
    rows = rows[:sample]
    tp = fp = fn = icorr = 0
    model.eval()
    for r in rows:
        pr = predict(r["text"], model, TOK, device, rev_i, rev_s)
        if pr["intent"] == r["intent"]:
            icorr += 1
        gold = {(k, normalize_slot_value(v)) for k, v in (r.get("slots") or {}).items()}
        pred = {(k, normalize_slot_value(v)) for k, v in (pr.get("slots") or {}).items()}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": len(rows),
        "intent_acc": round(icorr / max(1, len(rows)), 4),
        "slot_precision": round(prec, 4),
        "slot_recall": round(rec, 4),
        "slot_f1": round(f1, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--eval-sample", type=int, default=3000)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = resolve_device(None)

    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "datasets",
        "05_final_merged",
        "aviation_cmds_final_training_set.jsonl",
    )
    path = args.dataset or find_latest_version_path(base)
    with open(path, encoding="utf-8") as f:
        data = normalize_dataset([json.loads(line) for line in f if line.strip()])

    # Label space from the FULL dataset so every fraction shares one map.
    intents = sorted({r["intent"] for r in data})
    intent_map = {n: i for i, n in enumerate(intents)}
    slots = {"O"}
    for r in data:
        for s in r["slots"]:
            slots.add(f"B-{s}")
            slots.add(f"I-{s}")
    slot_map = {n: i for i, n in enumerate(sorted(slots))}

    train_full, val_data, test_data = split3(data)
    fractions = [float(x) for x in args.fractions.split(",")]
    print(f"device={device}  dataset={os.path.basename(path)}  rows={len(data)}")
    print(f"fixed val={len(val_data)}  fixed test={len(test_data)}  full train={len(train_full)}\n")

    results = []
    for frac in fractions:
        sub = subsample(train_full, frac)
        t0 = time.time()
        model = train_quick(
            sub,
            val_data,
            intent_map,
            slot_map,
            device,
            epochs=args.epochs,
            lr=args.lr,
            bs=args.batch_size,
            max_len=args.max_length,
            patience=args.patience,
        )
        metrics = eval_test(model, test_data, intent_map, slot_map, device, sample=args.eval_sample)
        row = {
            "fraction": frac,
            "train_rows": len(sub),
            "minutes": round((time.time() - t0) / 60, 1),
            **metrics,
        }
        results.append(row)
        print(
            f"frac {frac:>4}  train {len(sub):>6}  "
            f"intent {metrics['intent_acc']:.4f}  "
            f"slotF1 {metrics['slot_f1']:.4f}  "
            f"(P {metrics['slot_precision']:.3f} / R {metrics['slot_recall']:.3f})  "
            f"[{row['minutes']}m]"
        )
        del model

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "learning_curve.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(device),
                "dataset": os.path.basename(path),
                "eval_sample": args.eval_sample,
                "curve": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved -> {out}")
    print(
        "\nRead: if slot_f1 keeps rising to frac=1.0 -> more data helps; if flat -> diversity/quality is the wall."
    )


if __name__ == "__main__":
    main()
