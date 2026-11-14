"""Average trained models into one ("model soup"); see ML/utils/soup.py for why.

    python ML/average_models.py v22 v23 v24 --out v25 --holdout-from v24

Members and --out are version folders under ML/models/vimaan_nlu_model_best/
(--out may also be a path). The new folder gets the averaged weights, the
members' shared maps/config/tokenizer, the held-out files of --holdout-from
(default: the last member) for the gate, and a train_manifest.json naming the
members. Grade it with scripts/model_gate.py like any trained model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from utils.soup import SHARED_FILES, average, check_compatible

MODELS = Path(__file__).resolve().parent / "models" / "vimaan_nlu_model_best"
BODY = "model.safetensors"
HEAD = "intent_classifier.bin"
HOLDOUT_FILES = ("holdout_test.jsonl", "holdout_dev.jsonl")


def _mean(tensors: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    dtypes = {k: t.dtype for k, t in tensors[0].items()}
    avg = average([{k: t.float() for k, t in sd.items()} for sd in tensors])
    return {k: v.to(dtypes[k]).contiguous() for k, v in avg.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def build(members: list[Path], out: Path, holdout_from: Path) -> dict:
    check_compatible(members)
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    bodies = [load_file(m / BODY) for m in members]
    heads = [torch.load(m / HEAD, map_location="cpu", weights_only=True) for m in members]
    body, head = _mean(bodies), _mean(heads)

    out.mkdir(parents=True)
    save_file(body, out / BODY, metadata={"format": "pt"})
    torch.save(head, out / HEAD)
    for name in SHARED_FILES:
        shutil.copy2(members[0] / name, out / name)
    for name in HOLDOUT_FILES:
        if (holdout_from / name).exists():
            shutil.copy2(holdout_from / name, out / name)

    manifest = json.loads((holdout_from / "train_manifest.json").read_text())
    manifest.update(
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        soup_of=[m.name for m in members],
        soup_method="uniform average of every weight (body, slot head, intent head)",
        holdout_from=holdout_from.name,
        member_datasets={
            m.name: json.loads((m / "train_manifest.json").read_text()).get("dataset_path")
            for m in members
        },
        member_weights_sha256={m.name: _sha256(m / BODY) for m in members},
        note="Not trained: the average of the members. dataset_* and hyperparams "
        "are those of holdout_from.",
    )
    (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _resolve(name: str) -> Path:
    return Path(name) if "/" in name else MODELS / name


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("members", nargs="+", help="model versions to average, e.g. v22 v23 v24")
    p.add_argument("--out", required=True, help="new version (or path) to write")
    p.add_argument("--holdout-from", help="member whose held-out files go along (default: last)")
    args = p.parse_args(argv)

    members = [_resolve(m) for m in args.members]
    holdout_from = _resolve(args.holdout_from) if args.holdout_from else members[-1]
    if holdout_from not in members:
        p.error("--holdout-from must be one of the members")
    try:
        build(members, _resolve(args.out), holdout_from)
    except (ValueError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{' + '.join(m.name for m in members)} -> {_resolve(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
