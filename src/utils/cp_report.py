"""Final conformal evaluation of one arm from its per-split nonconformity scores.

Inputs
    scores   {split: [array per trace]} with splits `cal`, `test` and `external`
    y        {split: int array}
    alphas   target miscoverage levels

Outputs
    evaluate_arm()   per alpha: `test` (calibrate on `cal`, evaluate on `test`, with the Beta interval of coverage),
                     `external_recal` (repeated 50/50 splits of `external`), `external_zero_shot` (the `cal`
                     threshold applied to all of `external`)
    aggregate()      mean and standard deviation over seeds of every scalar metric
    to_markdown()    one table row per (arm, alpha)

CLI: evaluate the final seeds of one trained arm with the work package's score S = 1 - pi
    uv run python -m src.utils.cp_report --config C --out O --arm "<description of the arm>"
    reads <out>/final/seed<s>.npz and <out>/best.json; writes <out>/results.json and <out>/results.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.utils import cp_eval


def evaluate_arm(scores: dict, y: dict, alphas: list[float], n_recal: int = 200) -> dict:
    out = {}
    cal_label = cp_eval.label_scores(scores["cal"], y["cal"])
    for alpha in alphas:
        q = cp_eval.calibrate(cal_label, alpha)
        lo, hi = cp_eval.coverage_interval(len(cal_label), alpha)
        out[str(alpha)] = {
            "test": {"q": q, "coverage_interval": [lo, hi], **cp_eval.evaluate(scores["test"], y["test"], q)},
            "external_recal": cp_eval.recalibrated(scores["external"], y["external"], alpha, n_splits=n_recal),
            "external_zero_shot": cp_eval.evaluate(scores["external"], y["external"], q),
        }
    return out


def aggregate(per_seed: list[dict]) -> dict:
    """Mean and sd over seeds for every scalar leaf; lists are averaged element-wise."""
    first = per_seed[0]
    if isinstance(first, dict):
        return {k: aggregate([p[k] for p in per_seed]) for k in first}
    if isinstance(first, list):
        if any(v is None for p in per_seed for v in p):
            return None
        return np.mean(np.array(per_seed, dtype=float), axis=0).tolist()
    vals = np.array(per_seed, dtype=float)
    return {"mean": float(vals.mean()), "sd": float(vals.std())}


def to_markdown(arms: dict[str, dict], alphas: list[float]) -> str:
    """`arms` maps an arm description to its aggregated evaluate_arm() output."""
    def cell(d, k, digits=3):
        return f"{d[k]['mean']:.{digits}f} ± {d[k]['sd']:.{digits}f}"

    head = "| Arm | α | Test coverage | Test rel. size | Test size | Empty | Top-1 | External (recal.) coverage | External rel. size | External zero-shot coverage |\n"
    head += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    rows = []
    for name, res in arms.items():
        for alpha in alphas:
            r = res[str(alpha)]
            t, e, z = r["test"], r["external_recal"], r["external_zero_shot"]
            rows.append(
                f"| {name} | {alpha} | {cell(t, 'coverage')} | {cell(t, 'size_rel')} | {cell(t, 'size', 2)} | {cell(t, 'empty_rate')} "
                f"| {cell(t, 'top1')} | {cell(e, 'coverage')} | {cell(e, 'size_rel')} | {cell(z, 'coverage')} |"
            )
    return head + "\n".join(rows) + "\n"


def main() -> None:
    from src.utils.cp_sweep import load_logits

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    alphas = cfg["setup"]["metric"]["content"]["alphas"]
    runs = []
    for seed in cfg["params"]["seeds"]:
        logits = load_logits(args.out / "final" / f"seed{seed}.npz")
        scores = {s: [cp_eval.hps(cp_eval.softmax(l)) for l in d["logits"]] for s, d in logits.items()}
        runs.append(evaluate_arm(scores, {s: d["y"] for s, d in logits.items()}, alphas))
    results = {args.arm: aggregate(runs)}
    best = json.loads((args.out / "best.json").read_text())
    (args.out / "results.json").write_text(json.dumps({"selected": best["hp"], "arms": results}, indent=1))
    (args.out / "results.md").write_text(to_markdown(results, alphas))
    print(to_markdown(results, alphas))


if __name__ == "__main__":
    main()
