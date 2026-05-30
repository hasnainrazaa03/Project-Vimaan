"""Generate a `train_manifest.json` for an already-trained model directory.

Useful for backfilling provenance metadata when the model was trained
before manifests existed, or for re-stamping a manifest after a
dataset/git change.

Usage:
    python3 ML/data/generate_manifest.py \\
        --model-dir ML/models/vimaan_nlu_model_best/v9 \\
        --dataset   ML/datasets/05_final_merged/aviation_cmds_final_training_set.jsonl \\
        [--git-sha <sha>] [--max-length 64] [--epochs 10] [--lr 5e-5]

If --git-sha is omitted, the current `git rev-parse --short HEAD` is used.
Pass --force to overwrite an existing manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ML"))

from vimaan_nlu.manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    build_manifest,
    write_manifest,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--git-sha", default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest")
    args = parser.parse_args(argv)

    if not args.model_dir.is_dir():
        print(f"ERROR: --model-dir not found: {args.model_dir}", file=sys.stderr)
        return 2
    if not args.dataset.is_file():
        print(f"ERROR: --dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    out_path = args.model_dir / MANIFEST_FILENAME
    if out_path.exists() and not args.force:
        print(f"ERROR: {out_path} exists. Pass --force to overwrite.", file=sys.stderr)
        return 1

    rows = _read_jsonl(args.dataset)
    manifest = build_manifest(
        dataset_path=str(args.dataset),
        dataset_rows=rows,
        git_sha=args.git_sha,
        hyperparams={
            "max_length": args.max_length,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "base_model": args.base_model,
            "backfilled": True,
        },
    )
    written = write_manifest(args.model_dir, manifest)
    print(f"Wrote {written}")
    print(f"  dataset_sha256 = {manifest['dataset_sha256']}")
    print(f"  row_count      = {manifest['row_count']}")
    print(f"  intents        = {len(manifest['intent_counts'])}")
    print(f"  git_sha        = {manifest['git_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
