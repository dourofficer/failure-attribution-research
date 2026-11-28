"""Evaluate an unfinished sweep: pick the best finished run on validation and score its saved test logits.

Inputs
    --config   a resolved config (for the alphas of the metric setup)
    --out      one experiment folder holding `sweep/<tag>.json` and `sweep/<tag>.npz`
    --arm      description of the arm, used as the row label

Outputs
    <out>/partial_results.json   the selected run, how many runs were finished, and evaluate_arm() of its logits
    <out>/partial_results.md     one table row per alpha (single seed, so every "±" is 0)

The selection rule is the one of `cp_sweep select` (smallest validation relative set size), applied to the
runs on disk instead of the whole grid. Sweep runs use seed 0 only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.utils import cp_eval
from src.utils.cp_report import aggregate, evaluate_arm, to_markdown
from src.utils.cp_sweep import load_logits


def finished_runs(out: Path) -> list[dict]:
    """Metadata of every finished sweep run, best validation relative set size first."""
    runs = []
    for path in sorted((out / "sweep").glob("*.json")):
        meta = json.loads(path.read_text())
        runs.append({"tag": path.name[: -len(".json")], **meta})
    return sorted(runs, key=lambda r: (r["best_val"]["size_rel"], r["tag"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", type=str, required=True)
    args = ap.parse_args()
    alphas = yaml.safe_load(args.config.read_text())["setup"]["metric"]["content"]["alphas"]
    runs = finished_runs(args.out)
    best = runs[0]
    logits = load_logits(args.out / "sweep" / f"{best['tag']}.npz")
    scores = {s: [cp_eval.hps(cp_eval.softmax(l)) for l in d["logits"]] for s, d in logits.items()}
    res = {args.arm: aggregate([evaluate_arm(scores, {s: d["y"] for s, d in logits.items()}, alphas)])}
    meta = {"selected": best["hp"], "tag": best["tag"], "seed": best["seed"], "best_epoch": best["best_epoch"], "best_val": best["best_val"], "n_finished": len(runs)}
    (args.out / "partial_results.json").write_text(json.dumps({**meta, "arms": res}, indent=1))
    (args.out / "partial_results.md").write_text(to_markdown(res, alphas))
    print(f"{best['tag']} (best of {len(runs)} finished runs, epoch {best['best_epoch']})")
    print(to_markdown(res, alphas))


if __name__ == "__main__":
    main()
