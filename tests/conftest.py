"""Shared pytest fixtures and path setup.

Puts `ML/` on sys.path so tests can `import vimaan_nlu`, `import utils`,
etc. without each test repeating the boilerplate. This mirrors what the
plugin does at runtime, so we test the same import shape we ship.
"""

import os
import sys

import pytest

_ML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ML"))
if _ML_PATH not in sys.path:
    sys.path.insert(0, _ML_PATH)


def _torch_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def torch_available():
    return _torch_available()


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked `requires_torch` when torch isn't installed."""
    if _torch_available():
        return
    skip = pytest.mark.skip(reason="torch / transformers not installed")
    for item in items:
        if "requires_torch" in item.keywords:
            item.add_marker(skip)
