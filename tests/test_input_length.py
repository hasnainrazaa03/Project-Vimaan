"""Model input length (Tier 1 A7): truncate at the trained length, no padding."""

import json

import pytest

torch = pytest.importorskip("torch")
from vimaan_nlu.inference import DEFAULT_MAX_LENGTH, predict  # noqa: E402
from vimaan_nlu.model_loader import ModelLoader  # noqa: E402


@pytest.mark.parametrize(
    ("manifest", "length"),
    [
        ({"hyperparams": {"max_length": 40}}, 40),  # v17
        ({"hyperparams": {}}, DEFAULT_MAX_LENGTH),
        ({"hyperparams": {"max_length": "40"}}, DEFAULT_MAX_LENGTH),
        (None, DEFAULT_MAX_LENGTH),  # no manifest
    ],
)
def test_trained_length_comes_from_the_manifest(tmp_path, manifest, length):
    if manifest is not None:
        (tmp_path / "train_manifest.json").write_text(json.dumps(manifest))
    assert ModelLoader.trained_max_length(str(tmp_path)) == length


class Tokenizer:
    model_max_length = 40

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        n = len(text.split()) + 2
        return {"input_ids": torch.ones(1, n, dtype=torch.long), "attention_mask": torch.ones(1, n)}

    def convert_ids_to_tokens(self, ids):
        return ["[CLS]", *["x"] * (len(ids) - 2), "[SEP]"]


def model(input_ids, attention_mask):
    return None, torch.tensor([[5.0, 0.0]]), torch.zeros(1, input_ids.shape[1], 1)


def run(tokenizer, **kwargs):
    return predict(
        "gear up", model, tokenizer, torch.device("cpu"), {0: "None", 1: "x"}, {0: "O"}, **kwargs
    )


def test_one_utterance_is_not_padded_and_uses_the_trained_length():
    tokenizer = Tokenizer()
    run(tokenizer)
    assert tokenizer.calls[-1]["padding"] is False
    assert tokenizer.calls[-1]["max_length"] == 40


def test_old_fixed_padding_is_still_available():
    tokenizer = Tokenizer()
    run(tokenizer, max_length=32, pad_to_max_length=True)
    assert tokenizer.calls[-1]["padding"] == "max_length"
    assert tokenizer.calls[-1]["max_length"] == 32


def test_a_tokenizer_without_a_sane_limit_falls_back():
    tokenizer = Tokenizer()
    tokenizer.model_max_length = 10**30  # what Hugging Face reports when unset
    run(tokenizer)
    assert tokenizer.calls[-1]["max_length"] == DEFAULT_MAX_LENGTH
