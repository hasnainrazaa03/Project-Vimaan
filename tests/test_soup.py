"""Averaging trained models into one (ML/utils/soup.py, ML/average_models.py)."""

import json

import pytest
from utils.soup import SHARED_FILES, average, check_compatible


def model_dir(root, name, intents=("None", "set_flaps")):
    d = root / name
    d.mkdir()
    for f in SHARED_FILES:
        (d / f).write_text("{}")
    (d / "intent_map.json").write_text(json.dumps({i: n for n, i in enumerate(intents)}))
    return d


def test_average_is_the_mean_of_every_weight():
    out = average([{"a": 1.0, "b": 4.0}, {"a": 3.0, "b": 8.0}, {"a": 5.0, "b": 0.0}])
    assert out == {"a": 3.0, "b": 4.0}


def test_models_with_different_weights_are_refused():
    with pytest.raises(ValueError):
        average([{"a": 1.0}, {"a": 1.0, "b": 2.0}])


def test_members_must_share_labels_and_tokenizer(tmp_path):
    a, b = model_dir(tmp_path, "a"), model_dir(tmp_path, "b")
    check_compatible([a, b])
    c = model_dir(tmp_path, "c", intents=("set_flaps", "None"))
    with pytest.raises(ValueError, match="intent_map"):
        check_compatible([a, c])
    with pytest.raises(ValueError, match="at least two"):
        check_compatible([a])


@pytest.mark.requires_torch
def test_build_writes_an_averaged_model_the_gate_can_grade(tmp_path):
    import torch
    from average_models import BODY, HEAD, build
    from safetensors.torch import load_file, save_file

    members = []
    for n, value in (("v1", 1.0), ("v2", 2.0), ("v3", 6.0)):
        d = model_dir(tmp_path, n)
        save_file({"w": torch.full((2, 2), value)}, d / BODY)
        torch.save({"weight": torch.full((3,), value * 10)}, d / HEAD)
        (d / "train_manifest.json").write_text(json.dumps({"dataset_path": f"data_{n}"}))
        (d / "holdout_test.jsonl").write_text(n)
        members.append(d)

    out = tmp_path / "soup"
    manifest = build(members, out, holdout_from=members[1])
    assert torch.equal(load_file(out / BODY)["w"], torch.full((2, 2), 3.0))
    assert torch.equal(torch.load(out / HEAD, weights_only=True)["weight"], torch.full((3,), 30.0))
    assert (out / "holdout_test.jsonl").read_text() == "v2"
    assert all((out / f).exists() for f in SHARED_FILES)
    assert manifest["soup_of"] == ["v1", "v2", "v3"]
    assert manifest["member_datasets"] == {"v1": "data_v1", "v2": "data_v2", "v3": "data_v3"}
    assert json.loads((out / "train_manifest.json").read_text())["holdout_from"] == "v2"
    with pytest.raises(FileExistsError):
        build(members, out, holdout_from=members[1])
