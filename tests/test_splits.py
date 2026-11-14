"""How the trainer splits a v2 dataset (ML/utils/splits.py)."""

from utils.splits import template_split


def rows(*splits):
    return [{"text": f"t{i}", "split": s} for i, s in enumerate(splits)]


def test_dev_rows_are_the_validation_set_and_never_trained_on():
    data = rows("train", "dev", "test", "train", "dev")
    train, val, test = template_split(data)
    assert [r["text"] for r in train] == ["t0", "t3"]
    assert [r["text"] for r in val] == ["t1", "t4"]
    assert [r["text"] for r in test] == ["t2"]


def test_a_dataset_without_dev_rows_is_left_to_the_old_split():
    assert template_split(rows("train", "test", "train")) is None
    assert template_split([]) is None
