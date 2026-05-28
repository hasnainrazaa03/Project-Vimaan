"""Import-hygiene tests.

The purpose of these tests is to assert that lightweight consumers don't
pay the cost of loading torch + transformers just because they touched a
helper that lives in the `vimaan_nlu` package.

Regression target: B-009 (eager torch import via core/__init__.py).
"""

import subprocess
import sys


def _python_oneliner(snippet: str) -> str:
    """Run `snippet` in a clean python subprocess with ML/ on sys.path."""
    import os

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ml_path = os.path.join(repo_root, "ML")
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=ml_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"


def test_importing_vimaan_nlu_does_not_pull_torch():
    """Top-level `import vimaan_nlu` must NOT import torch transitively."""
    result = _python_oneliner(
        "import sys; sys.path.insert(0, '.');"
        "import vimaan_nlu;"
        "assert 'torch' not in sys.modules, 'torch was imported';"
        "print('OK')"
    )
    assert "OK" in result, result


def test_normalization_module_does_not_pull_torch():
    result = _python_oneliner(
        "import sys; sys.path.insert(0, '.');"
        "from vimaan_nlu.normalization import normalize_aviation_input;"
        "assert 'torch' not in sys.modules;"
        "print('OK')"
    )
    assert "OK" in result, result


def test_postprocessor_module_does_not_pull_torch():
    result = _python_oneliner(
        "import sys; sys.path.insert(0, '.');"
        "from vimaan_nlu.postprocessor import postprocess_slots;"
        "assert 'torch' not in sys.modules;"
        "print('OK')"
    )
    assert "OK" in result, result


def test_top_level_symbols_exported():
    """The vimaan_nlu package must publicly re-export the legacy symbols."""
    result = _python_oneliner(
        "import sys; sys.path.insert(0, '.');"
        "from vimaan_nlu import ("
        " normalize_aviation_input, normalize_dataset,"
        " JointIntentAndSlotModel, postprocess_slots, extract_numbers_from_text"
        ");"
        "print('OK')"
    )
    assert "OK" in result, result


def test_no_stale_core_imports_anywhere():
    """Source-code check: nothing under ML/ or plugin/ may still say `from core`."""
    import os
    import re

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pattern = re.compile(r"^\s*(from core[\s.]|import core[\s.])")
    offenders = []
    for sub in ("ML", "plugin"):
        for dirpath, _, filenames in os.walk(os.path.join(repo_root, sub)):
            if "__pycache__" in dirpath:
                continue
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                fp = os.path.join(dirpath, fn)
                with open(fp, encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        if pattern.match(line):
                            offenders.append(f"{fp}:{lineno}: {line.rstrip()}")
    assert not offenders, "Stale `core` imports found:\n" + "\n".join(offenders)
