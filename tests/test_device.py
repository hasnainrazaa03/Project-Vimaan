"""Tests for utils.device (MPS training support)."""

import os
import sys

import pytest

pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ML"))

import torch  # noqa: E402
from utils.device import pick_device, resolve_device  # noqa: E402


class TestPickDevice:
    def test_returns_a_torch_device(self):
        assert isinstance(pick_device(), torch.device)

    def test_disallowing_mps_never_returns_mps(self):
        assert pick_device(prefer_mps=False).type in ("cuda", "cpu")

    def test_resolve_explicit_cpu(self):
        assert resolve_device("cpu").type == "cpu"

    def test_resolve_auto_matches_pick(self):
        assert resolve_device(None).type == pick_device().type
        assert resolve_device("auto").type == pick_device().type
