"""Tests for ML/utils/file_utils.py — versioned path helpers.

These tests use a temporary directory and never touch the real models/
or datasets/ trees.
"""

import os

from utils.file_utils import (
    ensure_directory,
    find_latest_version_path,
    get_next_version_path,
)


class TestEnsureDirectory:
    def test_creates_missing(self, tmp_path):
        target = tmp_path / "newdir" / "child"
        out = ensure_directory(str(target))
        assert os.path.isdir(out)
        assert out == str(target)

    def test_idempotent(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        ensure_directory(str(target))
        ensure_directory(str(target))
        assert os.path.isdir(target)


class TestGetNextVersionPath:
    def test_starts_at_v1_when_empty(self, tmp_path):
        base = tmp_path / "dataset.jsonl"
        out = get_next_version_path(str(base))
        assert out.endswith("dataset_v1.jsonl")

    def test_increments_past_existing(self, tmp_path):
        for v in (1, 2, 3):
            (tmp_path / f"dataset_v{v}.jsonl").write_text("{}")
        out = get_next_version_path(str(tmp_path / "dataset.jsonl"))
        assert out.endswith("dataset_v4.jsonl")

    def test_skips_gaps(self, tmp_path):
        # v1 and v3 exist; loop advances until first unused slot — v2.
        (tmp_path / "dataset_v1.jsonl").write_text("{}")
        (tmp_path / "dataset_v3.jsonl").write_text("{}")
        out = get_next_version_path(str(tmp_path / "dataset.jsonl"))
        assert out.endswith("dataset_v2.jsonl")


class TestFindLatestVersionPath:
    def test_returns_highest_version(self, tmp_path):
        for v in (1, 2, 5):
            (tmp_path / f"dataset_v{v}.jsonl").write_text("{}")
        out = find_latest_version_path(str(tmp_path / "dataset.jsonl"))
        assert out is not None
        assert out.endswith("dataset_v5.jsonl")

    def test_missing_directory_returns_none(self, tmp_path):
        out = find_latest_version_path(str(tmp_path / "does_not_exist" / "x.jsonl"))
        assert out is None

    def test_falls_back_to_unversioned_when_present(self, tmp_path):
        base = tmp_path / "dataset.jsonl"
        base.write_text("{}")
        out = find_latest_version_path(str(base))
        assert out == str(base)

    def test_ignores_files_with_different_basename(self, tmp_path):
        # `other_v9.jsonl` must not be matched when probing `dataset`.
        (tmp_path / "other_v9.jsonl").write_text("{}")
        (tmp_path / "dataset_v2.jsonl").write_text("{}")
        out = find_latest_version_path(str(tmp_path / "dataset.jsonl"))
        assert out.endswith("dataset_v2.jsonl")
