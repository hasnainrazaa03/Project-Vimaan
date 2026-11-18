"""The joint model, inference and loader with an encoder other than
DistilBERT (q1 round 4: Ettin / ModernBERT for the question model). A tiny
random ModernBERT is built locally: no network, no downloaded weights."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ML"))

from vimaan_nlu.inference import extract_slots, extract_slots_by_offsets, is_wordpiece  # noqa: E402
from vimaan_nlu.model import JointIntentAndSlotModel  # noqa: E402

SLOTS = {0: "O", 1: "B-reading", 2: "I-reading", 3: "B-com_port"}


def test_slots_come_from_the_text_their_bpe_tokens_cover():
    text = "what's our vertical speed on com 2"
    offsets = [
        (0, 0),
        (0, 4),
        (4, 6),
        (6, 10),
        (10, 19),
        (19, 25),
        (25, 28),
        (28, 32),
        (32, 34),
        (0, 0),
    ]
    tags = [0, 0, 0, 0, 1, 2, 0, 0, 3, 0]
    assert extract_slots_by_offsets(tags, offsets, text, SLOTS) == {
        "reading": "vertical speed",
        "com_port": "2",
    }
    # An I- with no B- before it starts nothing, as in extract_slots.
    assert extract_slots_by_offsets([0, 0, 0, 0, 2, 2, 0, 0, 0, 0], offsets, text, SLOTS) == {}
    tokens = ["[CLS]", "what", "'", "Ġvertical"]
    assert extract_slots([0, 0, 0, 2], tokens, SLOTS) == {}


class _Backend:
    def __init__(self, kind):
        self.model = type(kind, (), {})()


class _Tok:
    def __init__(self, kind=None):
        if kind:
            self.backend_tokenizer = _Backend(kind)


def test_only_a_bpe_backend_takes_the_offsets_path():
    assert is_wordpiece(_Tok("WordPiece"))
    assert is_wordpiece(_Tok())  # no fast backend: as every model before
    assert not is_wordpiece(_Tok("BPE"))


def test_the_joint_model_runs_on_a_modernbert_encoder(tmp_path):
    config = transformers.ModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    transformers.ModernBertForMaskedLM(config).save_pretrained(tmp_path)
    model = JointIntentAndSlotModel(num_intents=3, num_slots=4, base_model=str(tmp_path))
    assert type(model.bert_for_slots).__name__ == "ModernBertForTokenClassification"
    assert model.encoder is model.bert_for_slots.model
    ids = torch.tensor([[1, 5, 6, 7, 2]])
    loss, intents, slots = model(
        ids, torch.ones_like(ids), torch.tensor([1]), torch.tensor([[-100, 0, 1, 2, -100]])
    )
    assert intents.shape == (1, 3) and slots.shape == (1, 5, 4) and loss.item() > 0
