"""Tests for scripts/install_plugin.py (R-11) — plugin installer planning."""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import install_plugin as ip  # noqa: E402


def _fake_xplane(tmp_path):
    root = tmp_path / "X-Plane 12"
    (root / "Resources" / "plugins" / "PythonPlugins").mkdir(parents=True)
    return root


def _fake_repo(tmp_path, versions=("v1", "v2", "v10")):
    repo = tmp_path / "repo"
    (repo / "plugin").mkdir(parents=True)
    (repo / "plugin" / "PI_VimaanCoPilot.py").write_text("# plugin\n")
    (repo / "ML" / "vimaan_nlu").mkdir(parents=True)
    (repo / "ML" / "utils").mkdir(parents=True)
    models = repo / "ML" / "models" / "vimaan_nlu_model_best"
    for v in versions:
        (models / v).mkdir(parents=True)
    return repo


class TestFindLatestModel:
    def test_none_when_missing(self, tmp_path):
        assert ip.find_latest_model(str(tmp_path / "nope")) is None

    def test_picks_highest_numerically(self, tmp_path):
        repo = _fake_repo(tmp_path)
        models = str(repo / "ML" / "models" / "vimaan_nlu_model_best")
        name, _ = ip.find_latest_model(models)
        assert name == "v10"  # not "v2" lexically

    def test_ignores_non_version_dirs(self, tmp_path):
        models = tmp_path / "m"
        (models / "archive").mkdir(parents=True)
        (models / "v3").mkdir()
        name, _ = ip.find_latest_model(str(models))
        assert name == "v3"


class TestPythonPluginsDir:
    def test_valid(self, tmp_path):
        xp = _fake_xplane(tmp_path)
        got = ip.python_plugins_dir(str(xp))
        assert got.endswith(os.path.join("plugins", "PythonPlugins"))

    def test_invalid_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ip.python_plugins_dir(str(tmp_path / "not-xplane"))


class TestBuildPlan:
    def test_targets_and_latest_model(self, tmp_path):
        repo, xp = _fake_repo(tmp_path), _fake_xplane(tmp_path)
        plan = ip.build_plan(str(repo), str(xp))
        by_name = {os.path.basename(d): d for _, d, _ in plan}
        assert set(by_name) >= {"PI_VimaanCoPilot.py", "vimaan_nlu", "utils", "v10"}
        # ML pieces land under PythonPlugins/ML, matching the plugin's resolution.
        assert by_name["vimaan_nlu"].endswith(os.path.join("PythonPlugins", "ML", "vimaan_nlu"))
        assert os.path.join("vimaan_nlu_model_best", "v10") in by_name["v10"]

    def test_specific_model(self, tmp_path):
        repo, xp = _fake_repo(tmp_path), _fake_xplane(tmp_path)
        plan = ip.build_plan(str(repo), str(xp), "v2")
        assert any(d.endswith(os.path.join("vimaan_nlu_model_best", "v2")) for _, d, _ in plan)

    def test_missing_model_raises(self, tmp_path):
        repo, xp = _fake_repo(tmp_path, versions=()), _fake_xplane(tmp_path)
        with pytest.raises(FileNotFoundError):
            ip.build_plan(str(repo), str(xp))

    def test_missing_specific_model_raises(self, tmp_path):
        repo, xp = _fake_repo(tmp_path), _fake_xplane(tmp_path)
        with pytest.raises(FileNotFoundError):
            ip.build_plan(str(repo), str(xp), "v99")

    def test_bad_xplane_raises(self, tmp_path):
        repo = _fake_repo(tmp_path)
        with pytest.raises(FileNotFoundError):
            ip.build_plan(str(repo), str(tmp_path / "nope"))


def test_main_dry_run_against_real_repo(tmp_path, capsys):
    # Uses the module's real REPO_ROOT (this checkout, which has a local model).
    rc = ip.main(["--xplane", str(_fake_xplane(tmp_path)), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "PI_VimaanCoPilot.py" in out
