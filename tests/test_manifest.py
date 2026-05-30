"""Tests for vimaan_nlu.manifest — provenance sidecar helpers."""

import json
import subprocess

import pytest
from vimaan_nlu.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    compute_dataset_sha256,
    compute_intent_counts,
    get_git_sha,
    load_manifest,
    write_manifest,
)


class TestComputeDatasetSha256:
    def test_deterministic(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_bytes(b'{"intent": "x"}\n{"intent": "y"}\n')
        a = compute_dataset_sha256(p)
        b = compute_dataset_sha256(p)
        assert a == b
        assert len(a) == 64  # sha256 hex

    def test_changes_with_content(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_bytes(b"a")
        sha_a = compute_dataset_sha256(p)
        p.write_bytes(b"b")
        sha_b = compute_dataset_sha256(p)
        assert sha_a != sha_b


class TestComputeIntentCounts:
    def test_counts_and_sorts(self):
        rows = [
            {"intent": "set_heading"},
            {"intent": "set_altitude"},
            {"intent": "set_heading"},
            {"intent": "set_heading"},
        ]
        out = compute_intent_counts(rows)
        assert out == {"set_altitude": 1, "set_heading": 3}
        # dict preserves insertion order; sorted means alphabetic keys.
        assert list(out.keys()) == sorted(out.keys())

    def test_skips_missing_intent(self):
        rows = [{"intent": "a"}, {"text": "no intent here"}, {"intent": "b"}]
        assert compute_intent_counts(rows) == {"a": 1, "b": 1}

    def test_empty(self):
        assert compute_intent_counts([]) == {}


class TestGetGitSha:
    def test_returns_string(self, tmp_path):
        # In the repo it returns a short SHA. In a non-git dir, "unknown".
        out = get_git_sha(tmp_path)
        assert isinstance(out, str)
        assert out == "unknown" or len(out) >= 4

    def test_unknown_on_missing_git(self, tmp_path, monkeypatch):
        def boom(*a, **kw):
            raise FileNotFoundError("git missing")

        monkeypatch.setattr(subprocess, "check_output", boom)
        assert get_git_sha(tmp_path) == "unknown"


class TestBuildManifest:
    def test_minimal_fields(self, tmp_path):
        ds = tmp_path / "ds.jsonl"
        ds.write_bytes(b'{"intent":"x"}\n')
        m = build_manifest(
            dataset_path=ds,
            dataset_rows=[{"intent": "x"}],
            hyperparams={"max_length": 32},
            git_sha="deadbeef",
        )
        assert m["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
        assert m["dataset_path"] == str(ds)
        assert m["row_count"] == 1
        assert m["intent_counts"] == {"x": 1}
        assert m["git_sha"] == "deadbeef"
        assert m["hyperparams"]["max_length"] == 32
        assert "framework_versions" in m
        assert "created_utc" in m

    def test_prefers_explicit_counts(self, tmp_path):
        ds = tmp_path / "ds.jsonl"
        ds.write_bytes(b"x")
        m = build_manifest(
            dataset_path=ds,
            row_count=999,
            intent_counts={"a": 1},
            git_sha="abc1234",
        )
        assert m["row_count"] == 999
        assert m["intent_counts"] == {"a": 1}

    def test_extra_payload_optional(self, tmp_path):
        ds = tmp_path / "ds.jsonl"
        ds.write_bytes(b"x")
        m = build_manifest(
            dataset_path=ds,
            row_count=0,
            intent_counts={},
            git_sha="x",
            extra={"note": "backfilled"},
        )
        assert m["extra"] == {"note": "backfilled"}


class TestWriteAndLoad:
    def test_roundtrip(self, tmp_path):
        ds = tmp_path / "ds.jsonl"
        ds.write_bytes(b'{"intent":"x"}\n')
        manifest = build_manifest(
            dataset_path=ds,
            dataset_rows=[{"intent": "x"}],
            git_sha="abc",
        )
        out = write_manifest(tmp_path / "model_v1", manifest)
        assert out.name == MANIFEST_FILENAME
        loaded = load_manifest(tmp_path / "model_v1")
        assert loaded == manifest

    def test_pretty_printed(self, tmp_path):
        ds = tmp_path / "ds.jsonl"
        ds.write_bytes(b"x")
        m = build_manifest(dataset_path=ds, row_count=0, intent_counts={}, git_sha="x")
        out = write_manifest(tmp_path / "m", m)
        text = out.read_text()
        assert "\n" in text  # indent=2 produces multi-line JSON
        assert text.endswith("\n")
        # Sanity: valid JSON
        json.loads(text)

    def test_creates_model_dir(self, tmp_path):
        ds = tmp_path / "ds.jsonl"
        ds.write_bytes(b"x")
        m = build_manifest(dataset_path=ds, row_count=0, intent_counts={}, git_sha="x")
        target = tmp_path / "nested" / "v1"
        assert not target.exists()
        write_manifest(target, m)
        assert (target / MANIFEST_FILENAME).is_file()


class TestGenerateManifestCLI:
    """Smoke-test the standalone backfill CLI end-to-end."""

    def test_cli_writes_manifest(self, tmp_path):
        ds = tmp_path / "ds.jsonl"
        ds.write_text('{"intent":"a"}\n{"intent":"b"}\n{"intent":"a"}\n')
        model_dir = tmp_path / "v1"
        model_dir.mkdir()

        from data import generate_manifest  # type: ignore  # noqa: PLC0415

        rc = generate_manifest.main(
            [
                "--model-dir",
                str(model_dir),
                "--dataset",
                str(ds),
                "--git-sha",
                "test1234",
                "--epochs",
                "5",
                "--max-length",
                "32",
            ]
        )
        assert rc == 0
        out = json.loads((model_dir / MANIFEST_FILENAME).read_text())
        assert out["git_sha"] == "test1234"
        assert out["row_count"] == 3
        assert out["intent_counts"] == {"a": 2, "b": 1}
        assert out["hyperparams"]["epochs"] == 5
        assert out["hyperparams"]["max_length"] == 32
        assert out["hyperparams"]["backfilled"] is True

    def test_cli_refuses_to_overwrite_without_force(self, tmp_path):
        ds = tmp_path / "ds.jsonl"
        ds.write_text('{"intent":"a"}\n')
        model_dir = tmp_path / "v1"
        model_dir.mkdir()
        (model_dir / MANIFEST_FILENAME).write_text("{}")

        from data import generate_manifest  # noqa: PLC0415

        rc = generate_manifest.main(
            ["--model-dir", str(model_dir), "--dataset", str(ds), "--git-sha", "x"]
        )
        assert rc == 1

    def test_cli_force_overwrites(self, tmp_path):
        ds = tmp_path / "ds.jsonl"
        ds.write_text('{"intent":"a"}\n')
        model_dir = tmp_path / "v1"
        model_dir.mkdir()
        (model_dir / MANIFEST_FILENAME).write_text("{}")

        from data import generate_manifest  # noqa: PLC0415

        rc = generate_manifest.main(
            [
                "--model-dir",
                str(model_dir),
                "--dataset",
                str(ds),
                "--git-sha",
                "x",
                "--force",
            ]
        )
        assert rc == 0
        out = json.loads((model_dir / MANIFEST_FILENAME).read_text())
        assert out["git_sha"] == "x"


@pytest.mark.parametrize("missing_arg", ["--model-dir", "--dataset"])
def test_cli_rejects_missing_paths(tmp_path, missing_arg):
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"intent":"a"}\n')
    model_dir = tmp_path / "v1"
    model_dir.mkdir()

    from data import generate_manifest  # noqa: PLC0415

    args = [
        "--model-dir",
        str(model_dir if missing_arg != "--model-dir" else tmp_path / "nope"),
        "--dataset",
        str(ds if missing_arg != "--dataset" else tmp_path / "nope.jsonl"),
        "--git-sha",
        "x",
    ]
    rc = generate_manifest.main(args)
    assert rc == 2
