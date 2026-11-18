"""Two fine-tuning methods meant to generalize past the training phrasings
(q1 round 4, 2026-11-17). Both are off unless asked for.

LP-FT (Kumar et al., ICLR 2022, "Fine-Tuning can Distort Pretrained Features
and Underperform Out-of-Distribution"): fit the new heads on the frozen
pretrained features first, then fine-tune everything. Starting from a good
head, fine-tuning moves the pretrained features less.

R-Drop (Liang et al., NeurIPS 2021): each batch goes through the model twice,
with two dropout masks, and the loss adds the symmetric KL between the two
predictions, so no single path through the network decides an answer.

The model is the joint model (vimaan_nlu.model.JointIntentAndSlotModel): an
encoder ``bert_for_slots.base_model``, a slot head ``bert_for_slots.classifier``
and an intent head ``intent_classifier``; ``model(ids, mask, intents, slots)``
returns (loss, intent logits, slot logits).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW


def heads(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [*model.intent_classifier.parameters(), *model.bert_for_slots.classifier.parameters()]


def sym_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Half of KL(p||q) + KL(q||p), averaged over the rows."""
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    return 0.5 * (
        F.kl_div(q, p, log_target=True, reduction="batchmean")
        + F.kl_div(p, q, log_target=True, reduction="batchmean")
    )


def _tensors(batch: dict[str, torch.Tensor], device) -> tuple[torch.Tensor, ...]:
    return tuple(
        batch[k].to(device) for k in ("input_ids", "attention_mask", "intent_label", "slot_labels")
    )


def rdrop_loss(
    model: torch.nn.Module, tensors: tuple[torch.Tensor, ...], alpha: float
) -> torch.Tensor:
    """The two passes' mean loss plus ``alpha`` times the symmetric KL of
    their intents and of their labelled tokens' slots."""
    ids, mask, intents, slots = tensors
    loss1, intent1, slot1 = model(ids, mask, intents, slots)
    loss2, intent2, slot2 = model(ids, mask, intents, slots)
    kl = sym_kl(intent1, intent2)
    labelled = slots != -100
    if labelled.any():
        kl = kl + sym_kl(slot1[labelled], slot2[labelled])
    return (loss1 + loss2) / 2 + alpha * kl


def linear_probe(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device,
    epochs: float,
    lr: float,
    trim: Callable[[dict[str, Any]], dict[str, Any]] = lambda b: b,
) -> float:
    """Train only the heads for ``epochs`` of ``loader`` on frozen features
    (the model in eval mode: no dropout). Leaves the model trainable and in
    train mode. Returns the mean loss."""
    encoder = model.bert_for_slots.base_model
    for p in encoder.parameters():
        p.requires_grad_(False)
    optimizer = AdamW(heads(model), lr=lr)
    steps = max(1, round(epochs * len(loader)))  # type: ignore[arg-type]
    model.eval()
    done, total = 0, 0.0
    try:
        while done < steps:
            for batch in loader:
                loss, _, _ = model(*_tensors(trim(batch), device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                done += 1
                if done >= steps:
                    break
    finally:
        for p in encoder.parameters():
            p.requires_grad_(True)
        model.train()
    return total / steps
