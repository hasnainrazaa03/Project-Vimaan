"""LP-FT and R-Drop (ML/utils/robust_ft.py) on a tiny stand-in for the joint model."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ML"))

from utils.robust_ft import heads, linear_probe, rdrop_loss, sym_kl  # noqa: E402


class Tiny(torch.nn.Module):
    """The joint model's shape: encoder, slot head, intent head."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.bert_for_slots = torch.nn.Module()
        self.bert_for_slots.base_model = torch.nn.Embedding(10, 8)
        self.bert_for_slots.classifier = torch.nn.Linear(8, 3)
        self.intent_classifier = torch.nn.Linear(8, 2)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, ids, mask, intents=None, slots=None):
        h = self.drop(self.bert_for_slots.base_model(ids))
        intent, slot = self.intent_classifier(h[:, 0]), self.bert_for_slots.classifier(h)
        ce = torch.nn.functional.cross_entropy
        loss = ce(intent, intents) + ce(slot.reshape(-1, 3), slots.reshape(-1), ignore_index=-100)
        return loss, intent, slot


def _batch():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 10, (4, 5), generator=g)
    slots = torch.randint(0, 3, (4, 5), generator=g)
    slots[:, 0] = -100
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "intent_label": torch.tensor([0, 1, 0, 1]),
        "slot_labels": slots,
    }


def test_sym_kl_is_zero_for_the_same_prediction_and_symmetric_otherwise():
    a, b = torch.tensor([[2.0, 0.0, -1.0]]), torch.tensor([[0.0, 1.0, 0.0]])
    assert float(sym_kl(a, a)) == pytest.approx(0.0, abs=1e-7)
    assert float(sym_kl(a, b)) > 0
    assert float(sym_kl(a, b)) == pytest.approx(float(sym_kl(b, a)))


def test_the_linear_probe_trains_only_the_heads_and_leaves_the_model_trainable():
    torch.manual_seed(0)
    model = Tiny()
    encoder = model.bert_for_slots.base_model.weight.detach().clone()
    head = [p.detach().clone() for p in heads(model)]
    loss = linear_probe(model, [_batch()] * 3, "cpu", epochs=2, lr=1e-2)
    assert loss > 0
    assert torch.equal(model.bert_for_slots.base_model.weight, encoder)
    assert all(not torch.equal(p, q) for p, q in zip(heads(model), head))
    assert all(p.requires_grad for p in model.parameters())
    assert model.training


def test_rdrop_adds_the_disagreement_of_two_dropout_passes():
    b = _batch()
    tensors = (b["input_ids"], b["attention_mask"], b["intent_label"], b["slot_labels"])
    model = Tiny(dropout=0.0)  # no dropout: both passes agree, nothing is added
    plain = model(*tensors)[0]
    assert rdrop_loss(model, tensors, alpha=1.0).item() == pytest.approx(plain.item())
    model = Tiny(dropout=0.5).train()
    torch.manual_seed(1)
    with_kl = rdrop_loss(model, tensors, alpha=1.0)
    torch.manual_seed(1)
    without = rdrop_loss(model, tensors, alpha=0.0)
    assert with_kl.item() > without.item()
