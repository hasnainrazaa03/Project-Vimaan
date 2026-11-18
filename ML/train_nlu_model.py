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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import dev_score, find_latest_version_path, get_model_versions_dir
from utils.convergence import Convergence
from utils.device import resolve_device
from utils.eval_metrics import per_intent_accuracy, per_slot_f1
from utils.lr_schedule import SCHEDULES, lr_factor
from utils.robust_ft import linear_probe, rdrop_loss
from utils.run_logger import RunLogger
from utils.slot_alignment import align_offsets_to_labels, find_slot_char_spans, trim_offsets
from utils.splits import template_split
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
        offsets = trim_offsets(text, encoding["offset_mapping"][0].tolist())

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
# The gradient norm costs a device sync, so it is sampled, not taken every step.
GRAD_NORM_EVERY = 25
# What a dev check picks the best weights by: dev loss, or dev scored the way
# the gate scores (utils/dev_score.py).
# "questions": a question model's dev (utils/dev_score.question_score).
SELECT = ("loss", "actions", "questions")
# With "wsd" and "actions": dev must gain more than this share of its rows
# (0.05%, ~4 of 7,708) to count as still improving.
MIN_GAIN_SHARE = 0.0005


def _dev_line(dev):
    if "misanswered" in dev:  # a question model (select="questions")
        return (
            f"  Dev questions right/misanswered/declined {dev['right']}/{dev['misanswered']}"
            f"/{dev['declined']} others answered {dev['others_answered']} score {dev['score']:g}"
        )
    return (
        f"  Dev right/rejected/wrong {dev['right']}/{dev['rejected']}/{dev['wrong']}"
        f" negatives acted {dev['negatives_acted']} score {dev['score']:g}"
    )


def _trim(batch):
    """Cut the batch's padding to its longest example. Rows are padded to
    max_length (40) but the longest averages 12 tokens; padding is masked, so
    the loss and outputs are the same (to ~1e-6) and a step is ~1.3x faster.
    Contiguous: on the CPU ``.to(device)`` keeps the slice, and the model's
    ``.view`` needs contiguous labels (a GPU copy hid this)."""
    n = int(batch["attention_mask"].sum(dim=1).max())
    return {k: v[:, :n].contiguous() if v.dim() == 2 else v for k, v in batch.items()}


def _optimizer(model, lr, device):
    """AdamW; fused into one kernel on a GPU (1.35x faster steps on an M5 GPU).
    The same update: fused and plain differ by ~0.01% of a step's change after
    30 steps, against ~0.0002% between two plain runs (MPS isn't deterministic)."""
    if device.type in ("cuda", "mps"):
        return AdamW(model.parameters(), lr=lr, fused=True), "AdamW (fused)"
    return AdamW(model.parameters(), lr=lr), "AdamW"


def _grads(model):
    return [p.grad for p in model.parameters() if p.grad is not None]


def _total_norm(grads):
    """L2 norm over all the gradients, as a tensor (no device sync). A sum of
    dot products: on the M5 GPU torch's norm takes 31 ms for the embedding's
    23M values and a dot product 0.4 ms, with the same result."""
    if not grads:
        return torch.zeros(())
    return torch.stack([torch.dot(g.reshape(-1), g.reshape(-1)) for g in grads]).sum().sqrt()


def _grad_norm(model):
    """L2 norm over every parameter's gradient. Only reads the gradients."""
    return float(_total_norm(_grads(model)))


def clip_grads_(model, max_norm):
    """Scale the gradients so their total norm is at most ``max_norm``: what
    torch.nn.utils.clip_grad_norm_ does (coefficient max_norm / (norm + 1e-6),
    at most 1), with the fast norm. torch's took a step from 25 to 7 per second
    on the M5 GPU. Returns the norm before clipping, as a tensor."""
    grads = _grads(model)
    total = _total_norm(grads)
    if grads:
        torch._foreach_mul_(grads, torch.clamp(max_norm / (total + 1e-6), max=1.0))
    return total


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
    runs_dir=None,
    schedule="constant",
    warmup=0.06,
    name=None,
    run_seed=None,
    clip_grad=0.0,
    eval_every=1.0,
    select="loss",
    decay_epochs=1.0,
    save_last=False,
    lp_epochs=0.0,
    lp_lr=1e-3,
    rdrop=0.0,
):
    # Reproducibility (T1.10): the pipeline was fully unseeded except the split.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading final dataset...")
    with open(dataset_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # v2 rows (ML/data/generate_v2.py) already carry slot values exactly as
    # they appear in the normalized text ("090", "121.5", "raise"), and say
    # which rows belong to held-out phrasings. Re-normalizing their values
    # would break that ("090" -> "90" no longer matches the text).
    v2 = bool(data) and all(r.get("v") == 2 for r in data)
    if not v2:
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

    by_template = template_split(data) if v2 else None
    if by_template:
        # Test = held-out templates (the gate's new phrasings); validation =
        # the dev templates, also never trained on (utils/splits.py).
        train_data, val_data, test_data = by_template
        validation = "dev templates"
    elif v2:
        # Test = rows from held-out templates (new phrasings); validation is a
        # random 15% of the rest.
        test_data = [r for r in data if r.get("split") == "test"]
        train_data, val_data = _split([r for r in data if r.get("split") != "test"], 0.15)
        validation = "random 15% of the training templates"
    else:
        train_data, temp_data = _split(data, 0.30)
        val_data, test_data = _split(temp_data, 0.50)
        validation = "random 15% of the rows"
    print(
        f"Split: train {len(train_data)}, val {len(val_data)} ({validation}), "
        f"held-out test {len(test_data)}"
    )

    intents = sorted(list(set(item["intent"] for item in data)))
    intent_map = {name: i for i, name in enumerate(intents)}

    slots = set(["O"])
    for item in data:
        for slot_name in item["slots"]:
            slots.add(f"B-{slot_name}")
            slots.add(f"I-{slot_name}")
    slot_map = {name: i for i, name in enumerate(sorted(list(slots)))}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
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
    model = JointIntentAndSlotModel(
        num_intents=len(intent_map), num_slots=len(slot_map), base_model=base_model
    ).to(device)
    if run_seed is not None:
        # Runs of one recipe to be averaged (ML/average_models.py): the same
        # starting weights (seeded above), a different data order and dropout.
        random.seed(42 + run_seed)
        np.random.seed(42 + run_seed)
        torch.manual_seed(42 + run_seed)
    if lp_epochs > 0:
        # LP-FT (utils/robust_ft.py): fit the heads on frozen features first.
        print(f"Linear probe: the heads alone for {lp_epochs:g} epochs at lr {lp_lr:g}...")
        probe_loss = linear_probe(model, train_loader, device, lp_epochs, lp_lr, trim=_trim)
        print(f"Linear probe done: mean loss {probe_loss:.4f}")
    optimizer, optimizer_name = _optimizer(model, lr, device)
    # "linear": warm up over the first `warmup` share of steps, then fall to
    # zero at the last step (utils/lr_schedule.py). Early stopping cuts the
    # decay short, so pair it with patience >= epochs.
    # "wsd": warm up, hold the rate while dev improves, decay once it stops
    # (utils/convergence.py); `num_epochs` is then a cap.
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, not {schedule!r}")
    if select not in SELECT:
        raise ValueError(f"select must be one of {SELECT}, not {select!r}")
    if not 0 < eval_every <= 1:
        raise ValueError("eval_every is a share of an epoch: more than 0, at most 1")
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup) if schedule in ("linear", "wsd") else 0
    eval_steps = max(1, round(eval_every * steps_per_epoch))
    convergence = None
    scheduler = None
    if schedule == "wsd":
        convergence = Convergence(
            patience=patience,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_steps=max(1, min(total_steps, round(decay_epochs * steps_per_epoch))),
            # Dev must gain more than this to count as still improving: a
            # check or two of dev rows going either way is noise.
            min_gain=MIN_GAIN_SHARE * len(val_data) if select != "loss" else 0.0,
        )
        scheduler = LambdaLR(optimizer, convergence.factor)
    elif schedule == "linear":
        scheduler = LambdaLR(optimizer, lambda step: lr_factor(step, total_steps, warmup_steps))

    best_val_loss = float("inf")  # dev loss at the best check
    best_score = None  # the selection score at the best check (higher is better)
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

    next_version = name or f"v{latest_version + 1}"
    model_save_path = os.path.join(models_dir, next_version)
    last_save_path = f"{model_save_path}-last"
    if name and os.path.exists(model_save_path):
        raise SystemExit(f"{model_save_path} already exists")
    if save_last and os.path.exists(last_save_path):
        raise SystemExit(f"{last_save_path} already exists")

    # Live metrics stream for the training monitor (ML/training_monitor.py).
    ml_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(runs_dir or os.path.join(ml_dir, "training_runs"), next_version)
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
        lr_schedule=schedule,
        warmup_steps=warmup_steps,
        patience=patience,
        optimizer=optimizer_name,
        clip_grad=clip_grad,
        eval_every=eval_every,
        select=select,
        decay_epochs=decay_epochs if schedule == "wsd" else None,
        lp_epochs=lp_epochs,
        lp_lr=lp_lr if lp_epochs > 0 else None,
        rdrop=rdrop,
        num_intents=len(intent_map),
        num_slots=len(slot_map),
        steps_per_epoch=steps_per_epoch,
        split={"train": len(train_data), "val": len(val_data), "test": len(test_data)},
        validation=validation,
        intent_dist=dict(Counter(r["intent"] for r in train_data).most_common()),
    )
    print(f"Live metrics -> {run_dir}/metrics.jsonl  (watch: python ML/training_monitor.py)")

    def hyperparams(at_epoch, val_loss):
        return {
            "max_length": max_length,
            "epochs": num_epochs,
            "lr": lr,
            "lr_schedule": schedule,
            "warmup_steps": warmup_steps,
            "batch_size": batch_size,
            "optimizer": optimizer_name,
            "clip_grad": clip_grad,
            "eval_every": eval_every,
            "select": select,
            "decay_epochs": decay_epochs if schedule == "wsd" else None,
            "lp_epochs": lp_epochs,
            "lp_lr": lp_lr if lp_epochs > 0 else None,
            "rdrop": rdrop,
            "early_stopping_patience": patience,
            "train_val_split": None if by_template else 0.85,
            "validation": validation,
            "random_state": 42,
            "run_seed": run_seed,
            "base_model": base_model,
            "best_val_loss": float(val_loss),
            "best_epoch": at_epoch,
        }

    def save(path, at_epoch, val_loss):
        os.makedirs(path, exist_ok=True)
        tokenizer.save_pretrained(path)
        model.bert_for_slots.save_pretrained(path)
        # .cpu() so the checkpoint is device-agnostic (e.g. saved from MPS).
        torch.save(
            {k: v.detach().cpu() for k, v in model.intent_classifier.state_dict().items()},
            f"{path}/intent_classifier.bin",
        )
        with open(f"{path}/intent_map.json", "w") as f:
            json.dump(intent_map, f)
        with open(f"{path}/slot_map.json", "w") as f:
            json.dump(slot_map, f)

        # Held-out test set (never trained or early-stopped on) so the
        # evaluator can measure true generalization — ML/evaluation/evaluator
        # loads this in preference to re-slicing the training file.
        with open(f"{path}/holdout_test.jsonl", "w", encoding="utf-8") as f:
            for row in test_data:
                f.write(json.dumps(row) + "\n")
        if by_template:  # the dev set, for comparing ideas without the test
            with open(f"{path}/holdout_dev.jsonl", "w", encoding="utf-8") as f:
                for row in val_data:
                    f.write(json.dumps(row) + "\n")

        # Provenance sidecar (Phase 3 / B-015 manifest). Written every time
        # the "best" model is overwritten so it always reflects the actually
        # checkpointed weights.
        manifest = build_manifest(
            dataset_path=dataset_path,
            dataset_rows=data,
            hyperparams=hyperparams(at_epoch, val_loss),
        )
        write_manifest(path, manifest)

    def evaluate(at_epoch):
        """Dev loss, accuracy and F1; with select="actions" also dev scored
        the way the gate scores (utils/dev_score.py)."""
        model.eval()
        total_val_loss = 0
        intent_correct = 0
        intent_total = 0
        intent_true = []
        intent_pred = []
        slot_true = []
        slot_pred = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {at_epoch:g} [Validation]"):
                batch = _trim(batch)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                intent_labels = batch["intent_label"].to(device)
                slot_labels = batch["slot_labels"].to(device)

                loss, intent_logits, slot_logits = model(
                    input_ids, attention_mask, intent_labels, slot_labels
                )
                total_val_loss += loss.item()

                predicted = intent_logits.argmax(dim=-1)
                intent_correct += (predicted == intent_labels).sum().item()
                intent_total += intent_labels.numel()
                intent_true.extend(intent_labels.cpu().tolist())
                intent_pred.extend(predicted.cpu().tolist())

                mask = slot_labels != -100
                slot_true.extend(slot_labels[mask].cpu().tolist())
                slot_pred.extend(slot_logits.argmax(dim=-1)[mask].cpu().tolist())
            dev = None
            if select != "loss":
                run = dev_score.live_runner(model, tokenizer, device, intent_map, slot_map)
                scorer = dev_score.question_score if select == "questions" else dev_score.score
                dev = scorer(run, val_data)
        model.train()
        val_slot_f1 = (
            f1_score(slot_true, slot_pred, average="macro", zero_division=0) if slot_true else 0.0
        )
        return {
            "val_loss": total_val_loss / len(val_loader),
            "val_intent_acc": intent_correct / max(intent_total, 1),
            "val_slot_f1": float(val_slot_f1),
            "per_intent": per_intent_accuracy(intent_true, intent_pred, intent_map),
            "per_slot": per_slot_f1(slot_true, slot_pred, slot_map),
            "dev": dev,
        }

    print("\nStarting training...")
    start_time = time.time()
    global_step = 0
    best_epoch = 0
    stopped_early = False
    last_eval = None  # (at_epoch, val_loss) of the latest check

    def _check(epoch, window_loss, window_steps):
        """One dev check: log it, save the best, decide whether to go on."""
        nonlocal best_val_loss, best_score, epochs_no_improve, best_epoch, last_eval
        at_epoch = round(global_step / steps_per_epoch, 2)
        if at_epoch == int(at_epoch):
            at_epoch = int(at_epoch)
        avg_train_loss = window_loss / max(window_steps, 1)
        print(f"Epoch {at_epoch:g} - Average Training Loss: {avg_train_loss:.4f}")
        val_started = time.time()
        logger.log(type="phase", phase="validate", epoch=epoch, elapsed=val_started - start_time)
        m = evaluate(at_epoch)
        dev = m["dev"]
        print(
            f"Epoch {at_epoch:g} - Val Loss: {m['val_loss']:.4f}  "
            f"Intent Acc: {m['val_intent_acc']:.4f}  Slot F1(macro): {m['val_slot_f1']:.4f}"
            + (_dev_line(dev) if dev else "")
        )
        score = dev["score"] if dev else -m["val_loss"]
        if convergence is not None:
            was_decaying = convergence.decaying
            is_best = convergence.check(global_step, score)
            if convergence.decaying and not was_decaying:
                print(
                    f"Dev has not improved for {patience} checks: converged at epoch "
                    f"{at_epoch:g}. Settling the rate to zero over {decay_epochs:g} epochs."
                )
        else:
            is_best = best_score is None or score > best_score
            epochs_no_improve = 0 if is_best else epochs_no_improve + 1
        logger.log(
            type="epoch",
            epoch=at_epoch,
            train_loss=avg_train_loss,
            val_loss=m["val_loss"],
            val_intent_acc=m["val_intent_acc"],
            val_slot_f1=m["val_slot_f1"],
            dev=dev,
            lr=optimizer.param_groups[0]["lr"],
            decaying=bool(convergence and convergence.decaying),
            best=bool(is_best),
            elapsed=time.time() - start_time,
            val_seconds=time.time() - val_started,
            epochs_no_improve=(
                convergence.checks_without_gain if convergence is not None else epochs_no_improve
            ),
            patience=patience,
            per_intent=m["per_intent"],
            per_slot=m["per_slot"],
        )
        last_eval = (at_epoch, m["val_loss"])
        if is_best:
            best_score = score
            best_val_loss = m["val_loss"]
            best_epoch = at_epoch
            logger.log(type="phase", phase="save", epoch=epoch, elapsed=time.time() - start_time)
            what = "Dev score" if dev else "Validation loss"
            print(f"{what} improved! Saving best model to {model_save_path}")
            save(model_save_path, at_epoch, m["val_loss"])
            return False
        if convergence is None:
            print(f"Validation loss did not improve. Count: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {at_epoch:g} epochs.")
                return True
        return False

    def _epochs():
        nonlocal global_step, stopped_early
        for epoch in range(1, num_epochs + 1):
            model.train()
            window_loss, window_steps = 0.0, 0
            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [Training]")):
                optimizer.zero_grad()
                batch = _trim(batch)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                intent_labels = batch["intent_label"].to(device)
                slot_labels = batch["slot_labels"].to(device)

                if rdrop > 0:  # utils/robust_ft.py: two dropout passes that must agree
                    loss = rdrop_loss(
                        model, (input_ids, attention_mask, intent_labels, slot_labels), rdrop
                    )
                else:
                    loss, _, _ = model(input_ids, attention_mask, intent_labels, slot_labels)

                step_loss = loss.item()
                window_loss += step_loss
                window_steps += 1
                loss.backward()
                global_step += 1
                extra = {}
                norm = clip_grads_(model, clip_grad) if clip_grad > 0 else None
                if global_step % GRAD_NORM_EVERY == 1:  # before clipping
                    extra["grad_norm"] = float(norm) if norm is not None else _grad_norm(model)
                optimizer.step()

                logger.log(
                    type="step",
                    epoch=epoch,
                    step=global_step,
                    loss=step_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    elapsed=time.time() - start_time,
                    **extra,
                )
                if scheduler is not None:
                    scheduler.step()  # after logging: the rate logged is the one used
                if convergence is not None:
                    convergence.step_done(global_step)
                finished = convergence is not None and convergence.finished(global_step)
                in_epoch = i + 1
                if in_epoch % eval_steps == 0 or in_epoch == steps_per_epoch or finished:
                    if _check(epoch, window_loss, window_steps):
                        stopped_early = True
                        return
                    window_loss, window_steps = 0.0, 0
                if finished:
                    return

    # A crash or Ctrl-C is recorded in the stream, so the dashboard shows
    # "failed" / "stopped" instead of a run that looks like it hung.
    try:
        _epochs()
        if save_last and last_eval is not None:
            print(f"Saving the last weights to {last_save_path}")
            save(last_save_path, *last_eval)
    except KeyboardInterrupt:
        logger.log(type="stopped", elapsed=time.time() - start_time)
        logger.close()
        raise
    except BaseException as exc:
        logger.log(
            type="error", message=f"{type(exc).__name__}: {exc}", elapsed=time.time() - start_time
        )
        logger.close()
        raise

    logger.log(
        type="done",
        best_val_loss=float(best_val_loss),
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        converged=convergence.reason if convergence is not None else None,
        decay_start_epoch=(
            round(convergence.decay_start / steps_per_epoch, 2)
            if convergence is not None and convergence.decaying
            else None
        ),
        elapsed=time.time() - start_time,
        model_path=model_save_path,
    )
    logger.close()
    print(
        f"\nTraining complete. Best at epoch {best_epoch:g} (val loss {best_val_loss:.4f})"
        f" -> {model_save_path}"
    )


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
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Dev checks without improvement before stopping (wsd: before the "
        "decay starts). A check comes every --eval-every of an epoch.",
    )
    parser.add_argument(
        "--schedule",
        choices=SCHEDULES,
        default="constant",
        help="Learning rate: constant; linear warmup then decay to zero; or wsd: "
        "warmup, steady until dev stops improving for --patience checks, then "
        "decay to zero over --decay-epochs and stop (--epochs is then a cap).",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.06,
        help="Share of all steps (of --epochs) spent warming up (linear and wsd).",
    )
    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=1.0,
        help="wsd: epochs over which the rate falls to zero once dev has converged.",
    )
    parser.add_argument(
        "--eval-every",
        type=float,
        default=1.0,
        help="Check dev every this share of an epoch (and at each epoch's end).",
    )
    parser.add_argument(
        "--select",
        choices=SELECT,
        default="loss",
        help="Keep the weights with the best dev loss, or the best dev score "
        "counted the way the gate counts (actions).",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=0.0,
        help="Clip the gradient norm to this (0: no clipping).",
    )
    parser.add_argument(
        "--save-last",
        action="store_true",
        help="Also save the final weights, as <name>-last.",
    )
    parser.add_argument(
        "--lp-epochs",
        type=float,
        default=0.0,
        help="LP-FT: first train only the heads on frozen features for this many epochs.",
    )
    parser.add_argument("--lp-lr", type=float, default=1e-3, help="LP-FT: the heads' rate.")
    parser.add_argument(
        "--rdrop",
        type=float,
        default=0.0,
        help="R-Drop: weight of the KL between two dropout passes (0: off; 2x the compute).",
    )
    parser.add_argument(
        "--model-save-dir",
        type=str,
        default=None,
        help="Override the parent dir for the new v<N> checkpoint.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name the run and its folder (e.g. v26-a) instead of the next v<N>. "
        "Not digits after the v, so nothing takes it for the latest model.",
    )
    parser.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help="Vary the data order and dropout (not the starting weights), for runs to be averaged.",
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
        name=args.name,
        run_seed=args.run_seed,
        schedule=args.schedule,
        warmup=args.warmup,
        clip_grad=args.clip_grad,
        eval_every=args.eval_every,
        select=args.select,
        decay_epochs=args.decay_epochs,
        save_last=args.save_last,
        lp_epochs=args.lp_epochs,
        lp_lr=args.lp_lr,
        rdrop=args.rdrop,
    )
