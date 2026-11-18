"""Average trained models into one ("model soup"); see ML/utils/soup.py for why.

    python ML/average_models.py v22 v23 v24 --out v25 --holdout-from v24
    python ML/average_models.py v26-j v26-k v26-l v26-m --greedy --out v26

--greedy chooses the members on dev (the "greedy soup"): the candidates are
ranked by their dev score, best first, and each is added only if the average
scores at least as well on dev as without it. Dev is the candidates' shared
holdout_dev.jsonl, scored the way the gate scores (utils/dev_score.py), never
the gate's test. A weak run (v26-g: dev loss 0.0127 against 0.0091 and 0.0096,
and lost "fly at ... thousand") is then left out by a rule set before the gate
is run, not picked out after it.

--within F averages every candidate whose own dev score is within F (a
fraction, e.g. 0.02) of the best one's, all with the same weight, the rest
left out. q1 attempt 2's greedy average was one run alone (q1-h, chosen by one
point of dev) and failed the gate on phrasings an average of two had passed;
an equal average of runs (v25) passed where single runs failed. The rule and
F are set before the runs are trained.

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
import tempfile
from collections.abc import Callable
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


def _dataset_sha(member: Path) -> str | None:
    return json.loads((member / "train_manifest.json").read_text()).get("dataset_sha256")


def is_question_model(path: Path) -> bool:
    """A question model (Tier 4, q1, q2...) knows no command it could act on."""
    from vimaan_nlu.commands import command

    intents = json.loads((path / "intent_map.json").read_text())
    return not any((c := command(i)) is not None and c.actionable for i in intents)


def _dev_scorer(dev_file: Path, questions: bool = False) -> Callable[[Path], dict]:
    """Dev scored like the gate: commands for a command model, questions
    (utils/dev_score.question_score) for a question model."""
    from utils import dev_score

    rows = [json.loads(line) for line in dev_file.read_text().splitlines() if line.strip()]
    scorer = dev_score.question_score if questions else dev_score.score
    return lambda path: scorer(dev_score.saved_runner(path), rows)


def _dev_score_for(
    candidates: list[Path],
    out: Path,
    holdout_from: Path,
    score: Callable[[Path], dict] | None,
) -> Callable[[Path], dict]:
    """The dev scorer for choosing among ``candidates`` (shared dev only)."""
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    if len({_dataset_sha(m) for m in candidates}) != 1:
        raise ValueError("choosing on dev needs candidates trained on the same dataset")
    dev_file = holdout_from / "holdout_dev.jsonl"
    if score is None:
        if not dev_file.exists():
            raise ValueError(f"{dev_file} is missing: choosing on dev needs a dev set")
        score = _dev_scorer(dev_file, is_question_model(candidates[0]))
    return score


def within(
    candidates: list[Path],
    out: Path,
    holdout_from: Path,
    fraction: float,
    score: Callable[[Path], dict] | None = None,
) -> dict:
    """Average, with equal weight, every candidate whose dev score is within
    ``fraction`` of the best one's; see the module docstring."""
    score = _dev_score_for(candidates, out, holdout_from, score)
    alone = {m: score(m) for m in candidates}
    best = max(s["score"] for s in alone.values())
    floor = best - fraction * abs(best)
    kept = [m for m in candidates if alone[m]["score"] >= floor]
    if holdout_from not in kept:
        holdout_from = kept[-1]
    if len(kept) == 1:
        shutil.copytree(kept[0], out)
        manifest = json.loads((out / "train_manifest.json").read_text())
        manifest.update(soup_of=[kept[0].name])
    else:
        manifest = build(kept, out, holdout_from)
    trail = [{"try": [m.name], "kept": m in kept, **alone[m]} for m in candidates]
    trail.append({"try": [m.name for m in kept], "kept": True, **score(out)})
    manifest.update(
        soup_method=f"every candidate within {fraction:g} of the best on dev, averaged "
        "with equal weight",
        soup_candidates={m.name: alone[m] for m in candidates},
        soup_trail=trail,
    )
    (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def greedy(
    candidates: list[Path],
    out: Path,
    holdout_from: Path,
    score: Callable[[Path], dict] | None = None,
) -> dict:
    """Average the candidates that help on dev; see the module docstring.
    ``score(path) -> {"score": ...}`` defaults to dev scored like the gate."""
    score = _dev_score_for(candidates, out, holdout_from, score)
    alone = {m: score(m) for m in candidates}
    order = sorted(candidates, key=lambda m: alone[m]["score"], reverse=True)
    kept, best = [order[0]], alone[order[0]]
    trail = [{"try": [order[0].name], "kept": True, **best}]
    with tempfile.TemporaryDirectory() as tmp:
        for i, m in enumerate(order[1:]):
            trial = Path(tmp) / f"trial{i}"
            build(kept + [m], trial, holdout_from)
            s = score(trial)
            keep = s["score"] >= best["score"]
            trail.append({"try": [k.name for k in kept] + [m.name], "kept": keep, **s})
            if keep:
                kept.append(m)
                best = s
            shutil.rmtree(trial)
    if len(kept) == 1:  # nothing averaged in helped: the best candidate as it is
        shutil.copytree(kept[0], out)
        manifest = json.loads((out / "train_manifest.json").read_text())
        manifest.update(soup_of=[kept[0].name])
    else:
        manifest = build(kept, out, holdout_from)
    manifest.update(
        soup_method="greedy on dev: candidates best-first, each kept only if the "
        "average scores at least as well on dev",
        soup_candidates={m.name: alone[m] for m in candidates},
        soup_trail=trail,
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
    p.add_argument("--greedy", action="store_true", help="choose the members on dev (see above)")
    p.add_argument(
        "--within",
        type=float,
        default=None,
        help="average every member within this fraction of the best on dev (see above)",
    )
    args = p.parse_args(argv)
    if args.greedy and args.within is not None:
        p.error("--greedy and --within are two ways to choose: pick one")

    members = [_resolve(m) for m in args.members]
    holdout_from = _resolve(args.holdout_from) if args.holdout_from else members[-1]
    if holdout_from not in members:
        p.error("--holdout-from must be one of the members")
    try:
        if args.greedy or args.within is not None:
            if args.greedy:
                manifest = greedy(members, _resolve(args.out), holdout_from)
            else:
                manifest = within(members, _resolve(args.out), holdout_from, args.within)
            for t in manifest["soup_trail"]:
                counts = (
                    f"questions right/misanswered/declined {t['right']}/{t['misanswered']}/"
                    f"{t['declined']} others answered {t['others_answered']}"
                    if "misanswered" in t
                    else f"right/rejected/wrong {t['right']}/{t['rejected']}/{t['wrong']} "
                    f"negatives acted {t['negatives_acted']}"
                )
                print(
                    f"  {' + '.join(t['try']):40s} dev {counts} score {t['score']:g} "
                    f"{'kept' if t['kept'] else 'left out'}"
                )
            members = [_resolve(m) for m in manifest["soup_of"]]
        else:
            build(members, _resolve(args.out), holdout_from)
    except (ValueError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{' + '.join(m.name for m in members)} -> {_resolve(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
