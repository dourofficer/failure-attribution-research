"""Hyperparameter sweep, selection and seeded final runs for a step scorer (one training mode per experiment).

Inputs
    config.resolved.yaml with params:
        emb_dir       the embedding cache (overridden by --emb, e.g. for smoke)
        fixed         hyperparameters shared by every run (mode, cls_splits, size_splits, batch_size, epochs, ...)
        grid          {name: [values]}; one run per combination, seed 0. May include `layer`.
        layer_from    optional path to another experiment's best.json; its `layer` is used when `grid` has none
        seeds         seeds of the final runs at the selected configuration
        eval_splits   splits whose logits are saved
    the cache itself (see cp_data)

Outputs under --out
    sweep/<tag>.json    hp, best validation metrics, best epoch, per-epoch history   (resume unit)
    sweep/<tag>.npz     logits at the best epoch: <split>_logits (concatenated), <split>_lengths, <split>_y
    best.json           the selected hp (lowest calibrated relative set size on validation at alpha) and the
                        validation table of every run
    final/seed<s>.json and .npz   the same two files for each final seed

CLI
    uv run python -m src.utils.cp_sweep run    --config C --out O --stage sweep|final --shard i --num-shards n [--emb E] [--layer-from P] [--limit k] [--epochs e]
    uv run python -m src.utils.cp_sweep select --config C --out O [--layer-from P] [--limit k]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.utils.cp_data import load_cache
from src.utils.cp_train import train


def grid_runs(params: dict, limit: int = 0) -> list[dict]:
    """Every combination of the grid, in a fixed order, merged over the fixed hyperparameters."""
    grid = dict(params.get("grid") or {})
    if "layer" not in grid:
        grid["layer"] = [json.loads(Path(params["layer_from"]).read_text())["hp"]["layer"]]
    names = sorted(grid)
    runs = [{**params["fixed"], **dict(zip(names, combo))} for combo in itertools.product(*(grid[n] for n in names))]
    for r in runs:
        r["_tag"] = "_".join(f"{n}{r[n]}" for n in names)
    return runs[:limit] if limit else runs


def save_run(out_stem: Path, hp: dict, seed: int, result: dict, splits: dict) -> None:
    arrays = {}
    for name, logits in result["logits"].items():
        arrays[f"{name}_logits"] = np.concatenate(logits).astype(np.float32)
        arrays[f"{name}_lengths"] = np.array([len(l) for l in logits], dtype=np.int32)
        arrays[f"{name}_y"] = np.array(splits[name].y, dtype=np.int32)
    # tags contain dots (lr0.0001), so suffixes are appended by hand and never via with_suffix
    tmp = out_stem.with_name("." + out_stem.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(out_stem.with_name(out_stem.name + ".npz"))
    meta = {"hp": {k: v for k, v in hp.items() if not k.startswith("_")}, "seed": seed, "best_epoch": result["best_epoch"], "best_val": result["best_val"], "history": result["history"]}
    out_stem.with_name(out_stem.name + ".json").write_text(json.dumps(meta, indent=1))   # written last: marks the run complete


def load_logits(path: Path) -> dict[str, dict]:
    """{split: {"logits": [array per trace], "y": array}} from a saved run."""
    z = np.load(path)
    out = {}
    for key in z.files:
        if key.endswith("_lengths"):
            name = key[: -len("_lengths")]
            bounds = np.concatenate([[0], np.cumsum(z[key])])
            flat = z[f"{name}_logits"]
            out[name] = {"logits": [flat[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)], "y": z[f"{name}_y"]}
    return out


def cmd_run(args, params: dict) -> None:
    emb_dir = Path(args.emb or params["emb_dir"])
    if args.stage == "sweep":
        jobs = [(r, 0, args.out / "sweep" / r["_tag"]) for r in grid_runs(params, args.limit)]
    else:
        best = json.loads((args.out / "best.json").read_text())["hp"]
        jobs = [(dict(best), s, args.out / "final" / f"seed{s}") for s in params["seeds"]]
    jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard and not j[2].with_name(j[2].name + ".json").exists()]
    if not jobs:
        print(f"{args.stage} shard {args.shard}: nothing to do")
        return
    (args.out / args.stage).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache: dict[int, dict] = {}
    for hp, seed, stem in tqdm(sorted(jobs, key=lambda j: j[0]["layer"]), desc=f"{args.stage} shard {args.shard}", unit="run"):
        if hp["layer"] not in cache:
            cache.clear()
            cache[hp["layer"]] = {k: v.to_device(device) for k, v in load_cache(emb_dir, hp["layer"]).items()}
        splits = cache[hp["layer"]]
        if args.epochs:
            hp = {**hp, "epochs": args.epochs}
        result = train(splits, hp, seed, params["eval_splits"])
        save_run(stem, hp, seed, result, splits)


def cmd_select(args, params: dict) -> None:
    runs = grid_runs(params, args.limit)
    table = []
    for r in runs:
        path = args.out / "sweep" / f"{r['_tag']}.json"
        if not path.exists():
            raise SystemExit(f"missing sweep run {path}")
        meta = json.loads(path.read_text())
        table.append({"tag": r["_tag"], "hp": meta["hp"], "best_epoch": meta["best_epoch"], **meta["best_val"]})
    table.sort(key=lambda t: (t["size_rel"], t["tag"]))
    (args.out / "best.json").write_text(json.dumps({"hp": table[0]["hp"], "tag": table[0]["tag"], "n_runs": len(table), "table": table}, indent=1))
    print(f"selected {table[0]['tag']} of {len(table)} runs: val size_rel={table[0]['size_rel']:.4f} coverage={table[0]['coverage']:.3f} top1={table[0]['top1']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "select"])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--emb", type=str, default="")
    ap.add_argument("--stage", choices=["sweep", "final"], default="sweep")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--layer-from", type=str, default="", help="smoke: override params.layer_from")
    ap.add_argument("--limit", type=int, default=0, help="smoke: keep only the first k grid runs")
    ap.add_argument("--epochs", type=int, default=0, help="smoke: override the number of epochs")
    args = ap.parse_args()
    params = yaml.safe_load(args.config.read_text())["params"]
    if args.layer_from:
        params["layer_from"] = args.layer_from
    (cmd_run if args.cmd == "run" else cmd_select)(args, params)


if __name__ == "__main__":
    main()
