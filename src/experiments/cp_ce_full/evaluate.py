"""Final evaluation of the cross-entropy reference scorer, its no-retraining arms and the floors.

Inputs
    config.resolved.yaml (metric setup: alphas; params.posthoc; params.seeds)
    <out>/final/seed<s>.npz   logits of the selected configuration, one file per seed
    the embedding cache (labels and lengths of the training splits, for the position prior)

Outputs
    <out>/results.json   per arm: mean and sd over seeds of every metric, per alpha; the settings each tuned arm chose
    <out>/results.md     the headline table

Usage
    uv run python -m src.experiments.cp_ce_full.evaluate --config C --out O [--emb E]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.experiments.cp_ce_full.posthoc import arms
from src.utils.cp_data import load_cache
from src.utils.cp_report import aggregate, evaluate_arm, to_markdown
from src.utils.cp_sweep import load_logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--emb", type=str, default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    params, metric = cfg["params"], cfg["setup"]["metric"]["content"]
    alphas, sel_alpha = metric["alphas"], metric["selection"]["alpha"]
    best = json.loads((args.out / "best.json").read_text())
    cache = load_cache(Path(args.emb or params["emb_dir"]), best["hp"]["layer"])
    train_rel_pos = np.concatenate([np.array(cache[s].y) / cache[s].lengths for s in params["fixed"]["cls_splits"]])

    per_arm: dict[str, list] = {}
    chosen = []
    for seed in params["seeds"]:
        logits = load_logits(args.out / "final" / f"seed{seed}.npz")
        y = {s: d["y"] for s, d in logits.items()}
        seed_arms, picked = arms(logits, train_rel_pos, params["posthoc"], sel_alpha, seed)
        chosen.append(picked)
        for name, scores in seed_arms.items():
            per_arm.setdefault(name, []).append(evaluate_arm(scores, y, alphas))
    results = {name: aggregate(runs) for name, runs in per_arm.items()}
    (args.out / "results.json").write_text(json.dumps({"selected": best["hp"], "chosen_per_seed": chosen, "arms": results}, indent=1))
    (args.out / "results.md").write_text(to_markdown(results, alphas))
    print(to_markdown(results, alphas))


if __name__ == "__main__":
    main()
