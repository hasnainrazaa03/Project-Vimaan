"""Install the Vimaan plugin into an X-Plane installation (R-11).

Copies the plugin entry point, the runtime ML package, and the latest trained
model into ``<X-Plane>/Resources/plugins/PythonPlugins/`` — where XPPython3
loads plugins and where the plugin resolves its ML package (``PythonPlugins/ML``).
Only the runtime pieces are copied (``vimaan_nlu`` + ``utils`` + one model
version) — not datasets, all model versions, or dev tooling.

Usage::

    python scripts/install_plugin.py --xplane "/path/to/X-Plane 12"
    python scripts/install_plugin.py --xplane "..." --dry-run
    python scripts/install_plugin.py --xplane "..." --model v10
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODELS_SUBDIR = os.path.join("ML", "models", "vimaan_nlu_model_best")


def find_latest_model(models_root: str):
    """Return ``(name, path)`` of the highest ``vN`` under ``models_root``, else None."""
    if not os.path.isdir(models_root):
        return None
    best = None
    for name in os.listdir(models_root):
        if name.startswith("v") and name[1:].isdigit():
            num = int(name[1:])
            if best is None or num > best[0]:
                best = (num, name)
    if best is None:
        return None
    return best[1], os.path.join(models_root, best[1])


def python_plugins_dir(xplane_root: str) -> str:
    """Validate an X-Plane root and return its ``PythonPlugins`` directory."""
    plugins = os.path.join(xplane_root, "Resources", "plugins")
    if not os.path.isdir(plugins):
        raise FileNotFoundError(
            f"not an X-Plane installation (no Resources/plugins): {xplane_root}"
        )
    return os.path.join(plugins, "PythonPlugins")


def build_plan(repo_root: str, xplane_root: str, model_version: str | None = None):
    """Return a list of ``(src, dst, kind)`` copy ops (``kind`` in {"file","tree"}).

    Raises ``FileNotFoundError`` if the X-Plane root or a trained model is missing.
    """
    dest_root = python_plugins_dir(xplane_root)
    ml_dest = os.path.join(dest_root, "ML")
    models_root = os.path.join(repo_root, MODELS_SUBDIR)

    if model_version:
        model_name = model_version
        model_src = os.path.join(models_root, model_version)
        if not os.path.isdir(model_src):
            raise FileNotFoundError(f"model not found: {model_src}")
    else:
        latest = find_latest_model(models_root)
        if latest is None:
            raise FileNotFoundError(
                f"no trained model under {models_root} — train one or run scripts/fetch_model.sh"
            )
        model_name, model_src = latest

    return [
        (
            os.path.join(repo_root, "plugin", "PI_VimaanCoPilot.py"),
            os.path.join(dest_root, "PI_VimaanCoPilot.py"),
            "file",
        ),
        (os.path.join(repo_root, "ML", "vimaan_nlu"), os.path.join(ml_dest, "vimaan_nlu"), "tree"),
        (os.path.join(repo_root, "ML", "utils"), os.path.join(ml_dest, "utils"), "tree"),
        (
            model_src,
            os.path.join(ml_dest, "models", "vimaan_nlu_model_best", model_name),
            "tree",
        ),
    ]


def _copy(src: str, dst: str, kind: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if kind == "tree":
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Install the Vimaan plugin into X-Plane.")
    parser.add_argument("--xplane", required=True, help="X-Plane root (contains Resources/)")
    parser.add_argument("--model", default=None, help="model version, e.g. v10 (default: latest)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without copying")
    args = parser.parse_args(argv)

    try:
        plan = build_plan(REPO_ROOT, args.xplane, args.model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    dest = python_plugins_dir(args.xplane)
    print(f"{'DRY RUN — ' if args.dry_run else ''}installing into {dest}\n")
    for src, dst, kind in plan:
        print(f"  {kind:4s}  {os.path.relpath(src, REPO_ROOT)}  ->  {dst}")
        if not args.dry_run:
            if not os.path.exists(src):
                print(f"ERROR: missing source: {src}", file=sys.stderr)
                return 2
            _copy(src, dst, kind)

    if not args.dry_run:
        print("\nDone. Start X-Plane and check Log.txt for '[Vimaan] Model loaded'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
