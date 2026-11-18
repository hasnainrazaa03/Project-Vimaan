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


def _weighted_models(tmp_path, values, dataset="sha"):
    import torch
    from average_models import BODY, HEAD
    from safetensors.torch import save_file

    out = []
    for n, value in values.items():
        d = model_dir(tmp_path, n)
        save_file({"w": torch.full((1,), value)}, d / BODY)
        torch.save({"weight": torch.full((1,), value)}, d / HEAD)
        (d / "train_manifest.json").write_text(json.dumps({"dataset_sha256": dataset}))
        (d / "holdout_dev.jsonl").write_text("")
        out.append(d)
    return out


def _closeness_to(target):
    """A dev score that is best when the model's weight is nearest ``target``."""
    from average_models import BODY
    from safetensors.torch import load_file

    def score(path):
        w = load_file(path / BODY)["w"].item()
        return {
            "score": -abs(w - target),
            "right": 0,
            "rejected": 0,
            "wrong": 0,
            "negatives_acted": 0,
        }

    return score


@pytest.mark.requires_torch
def test_greedy_leaves_out_a_member_that_makes_dev_worse(tmp_path):
    from average_models import greedy

    a, b, c = _weighted_models(tmp_path, {"a": 4.0, "b": 6.0, "g": 40.0})
    manifest = greedy([c, a, b], tmp_path / "soup", a, score=_closeness_to(5.0))
    assert manifest["soup_of"] == ["a", "b"]  # best-first; g would drag it to 16.7
    assert [t["kept"] for t in manifest["soup_trail"]] == [True, True, False]
    assert set(manifest["soup_candidates"]) == {"a", "b", "g"}
    assert "greedy" in manifest["soup_method"]


@pytest.mark.requires_torch
def test_greedy_keeps_the_best_alone_when_averaging_never_helps(tmp_path):
    from average_models import BODY, greedy
    from safetensors.torch import load_file

    a, b = _weighted_models(tmp_path, {"a": 5.0, "b": 9.0})
    manifest = greedy([a, b], tmp_path / "soup", a, score=_closeness_to(5.0))
    assert manifest["soup_of"] == ["a"]
    assert load_file(tmp_path / "soup" / BODY)["w"].item() == 5.0


@pytest.mark.requires_torch
def test_greedy_needs_one_dev_set(tmp_path):
    from average_models import greedy

    (a,) = _weighted_models(tmp_path, {"a": 1.0})
    (b,) = _weighted_models(tmp_path, {"b": 1.0}, dataset="other")
    with pytest.raises(ValueError, match="same dataset"):
        greedy([a, b], tmp_path / "soup", a, score=_closeness_to(1.0))


def _fixed(scores):
    """A dev score per member name (the average gets ``scores["soup"]``)."""

    def score(path):
        return {
            "score": scores.get(path.name, scores["soup"]),
            "right": 0,
            "rejected": 0,
            "wrong": 0,
            "negatives_acted": 0,
        }

    return score


@pytest.mark.requires_torch
def test_within_averages_every_member_near_the_best_with_equal_weight(tmp_path):
    from average_models import BODY, within
    from safetensors.torch import load_file

    a, b, c, d = _weighted_models(tmp_path, {"a": 1.0, "b": 2.0, "c": 3.0, "d": 40.0})
    # d is more than 2% below the best (100): left out; c is 1.5% below: kept.
    scores = {"a": 100.0, "b": 99.0, "c": 98.5, "d": 97.9, "soup": 99.5}
    manifest = within([a, b, c, d], tmp_path / "soup", d, 0.02, score=_fixed(scores))
    assert manifest["soup_of"] == ["a", "b", "c"]
    assert load_file(tmp_path / "soup" / BODY)["w"].item() == pytest.approx(2.0)
    assert [t["kept"] for t in manifest["soup_trail"]] == [True, True, True, False, True]
    assert manifest["soup_trail"][-1]["score"] == 99.5
    assert "within 0.02" in manifest["soup_method"]


@pytest.mark.requires_torch
def test_within_needs_one_dev_set(tmp_path):
    from average_models import within

    (a,) = _weighted_models(tmp_path, {"a": 1.0})
    (b,) = _weighted_models(tmp_path, {"b": 1.0}, dataset="other")
    with pytest.raises(ValueError, match="same dataset"):
        within([a, b], tmp_path / "soup", a, 0.02, score=_fixed({"soup": 0.0}))


@pytest.mark.requires_torch
def test_a_question_model_is_told_from_a_command_model(tmp_path):
    from average_models import is_question_model

    assert is_question_model(model_dir(tmp_path, "q1-a", ("None", "ask_fuel", "ask_time")))
    assert not is_question_model(model_dir(tmp_path, "v27-a", ("None", "ask_fuel", "set_flaps")))
