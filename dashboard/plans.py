"""Training plans: a queue of terminal runs shown as one card, so a round of
many runs (candidates, each a few seeds) can be followed from the dashboard.

A plan is a JSON file in ``ML/training_plans/`` (not in git), written by
whoever queues the runs::

    {"title": "q1 round 4",
     "note": "what is being compared, how the winner is picked",
     "groups": [{"name": "LP-FT", "runs": ["q1-lp5", "q1-lp6"],
                 "cost": 1, "note": "...", "result": "..."}]}

Each run's state comes from its metrics stream (dashboard/run_metrics.py); a
run with no stream yet is "queued". Time left is an estimate: a queued run
takes as long as its group's finished runs did (else the plan's, times the
group's ``cost``: 2 for a run that does twice the work per step).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from dashboard import paths, run_metrics

_PLAN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_DONE = {"finished", "failed", "stopped", "interrupted"}


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _run_row(name: str) -> dict[str, Any]:
    s = run_metrics.summarize(name) if _PLAN_NAME.match(name) else None
    if s is None:
        return {"run": name, "status": "queued"}
    dev = s.get("best_dev") or {}
    return {
        "run": name,
        "status": s["status"],
        "phase": s.get("phase"),
        "epoch": s.get("epoch"),
        "best_epoch": s.get("best_epoch"),
        "best_score": dev.get("score") if isinstance(dev, dict) else None,
        "elapsed": s.get("elapsed"),
        "eta": s.get("eta_soonest") or s.get("eta"),
        "updated": s.get("updated"),
    }


def plan_status(plan: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    """The plan with each run's state, per-group counts and an estimate of
    the time left."""
    now = time.time() if now is None else now
    groups = []
    for g in plan.get("groups") or []:
        cost = float(g.get("cost") or 1.0)
        rows = [_run_row(str(r)) for r in g.get("runs") or []]
        groups.append({**g, "cost": cost, "runs": rows})

    # A plain run's time: finished runs' time over their group's cost.
    base = _mean(
        [
            r["elapsed"] / g["cost"]
            for g in groups
            for r in g["runs"]
            if r["status"] == "finished" and r.get("elapsed")
        ]
    )
    left = 0.0
    unknown = False
    for g in groups:
        mean = _mean(
            [r["elapsed"] for r in g["runs"] if r["status"] == "finished" and r.get("elapsed")]
        )
        each = mean if mean is not None else (base * g["cost"] if base is not None else None)
        g["done"] = sum(r["status"] in _DONE for r in g["runs"])
        g["each_seconds"] = each
        for r in g["runs"]:
            if r["status"] == "running":
                if each is not None:
                    r["left"] = max(each - (r.get("elapsed") or 0.0), 60.0)
                else:
                    r["left"] = r.get("eta")
            elif r["status"] == "queued":
                r["left"] = each
            else:
                continue
            if r["left"] is None:
                unknown = True
            else:
                left += r["left"]
    runs = [r for g in groups for r in g["runs"]]
    return {
        "title": plan.get("title"),
        "note": plan.get("note"),
        "groups": groups,
        "total": len(runs),
        "done": sum(r["status"] in _DONE for r in runs),
        "failed": [r["run"] for r in runs if r["status"] in ("failed", "stopped", "interrupted")],
        "running": [r["run"] for r in runs if r["status"] == "running"],
        "left_seconds": None if unknown else left,
        "ends_at": None if unknown or not left else now + left,
    }


def list_plans() -> list[dict[str, Any]]:
    """Every plan, the most recently written first; a file that isn't a plan
    is listed with its error."""
    d = paths.TRAINING_PLANS
    if not d.is_dir():
        return []
    files = [f for f in d.glob("*.json") if _PLAN_NAME.match(f.stem)]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    out = []
    for f in files:
        try:
            plan = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(plan, dict) or not isinstance(plan.get("groups"), list):
                raise ValueError("a plan needs a list of groups")
        except (OSError, ValueError) as exc:
            out.append({"name": f.stem, "error": str(exc)})
            continue
        out.append({"name": f.stem, **plan_status(plan)})
    return out
