"""Build task-disjoint splits of failed, first-error-labelled traces.

Inputs
    config.resolved.yaml   the dataset setups (path, filter, seed, splits) and params.smoke_per_split
    the normalized JSONL files the dataset setups point to

Outputs
    <out>/<dataset_id>.jsonl   one record per kept task: id, task_id, dataset, subset, split, prompt, steps,
                               num_steps, y (first error step, 0-indexed)
    <out>/summary.json         counts per dataset: seen, dropped per reason, tasks, tasks per split

Usage
    uv run python -m src.experiments.cp_step_embeddings.splits --config <resolved> --out <dir> [--smoke]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import yaml
from tqdm import tqdm


def _key(seed: int, text: str) -> str:
    """Seeded, order-free sort key: the same task always lands in the same place."""
    return hashlib.sha256(f"{seed}|{text}".encode()).hexdigest()


def keep(rec: dict, flt: dict) -> str | None:
    """Return None if the record passes the filter, else the reason it is dropped."""
    if flt.get("has_error") and not rec["has_error"]:
        return "no_error"
    n = rec["num_steps"]
    if n < flt["min_steps"]:
        return "too_short"
    if flt.get("max_steps") is not None and n > flt["max_steps"]:
        return "too_long"
    if flt.get("drop_empty_steps") and any(not s.strip() for s in rec["steps"]):
        return "empty_step"
    return None


def assign_splits(records: Iterable[dict], setup: dict) -> tuple[list[dict], Counter]:
    """Filter, keep one trace per task, and assign tasks to splits.

    Tasks are ordered by a seeded hash and cut in the order the setup lists its splits. A split size of
    `all` takes every remaining task. Tasks beyond the listed sizes are left out (reserve).
    """
    seed, flt = setup["seed"], setup["filter"]
    stats: Counter = Counter()
    best: dict[str, dict] = {}
    for rec in records:
        stats["seen"] += 1
        reason = keep(rec, flt)
        if reason is not None:
            stats[f"dropped_{reason}"] += 1
            continue
        stats["kept_traces"] += 1
        tid = rec[setup["group_key"]]
        cur = best.get(tid)
        if cur is None or _key(seed, rec["id"]) < _key(seed, cur["id"]):
            best[tid] = rec
    tasks = sorted(best, key=lambda t: _key(seed, t))
    stats["tasks"] = len(tasks)

    out, start = [], 0
    for split, size in setup["splits"].items():
        end = len(tasks) if size == "all" else start + int(size)
        if end > len(tasks):
            raise ValueError(f"split {split} needs tasks up to {end}, only {len(tasks)} available")
        for tid in tasks[start:end]:
            r = best[tid]
            out.append(
                {
                    "id": r["id"],
                    "task_id": r["task_id"],
                    "dataset": r["dataset"],
                    "subset": r["subset"],
                    "split": split,
                    "prompt": r["prompt"],
                    "steps": r["steps"],
                    "num_steps": r["num_steps"],
                    "y": r[setup["label"]],
                }
            )
        stats[f"split_{split}"] = end - start
        start = end
    stats["reserve_tasks"] = len(tasks) - start
    return out, stats


def take_smoke(rows: list[dict], per_split: int) -> list[dict]:
    """Keep the first `per_split` tasks of every split, in order."""
    seen: Counter = Counter()
    out = []
    for r in rows:
        if seen[r["split"]] < per_split:
            seen[r["split"]] += 1
            out.append(r)
    return out


def _stream(path: Path, desc: str) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, unit="rows"):
            # cheap prefilter: all setups keep failed traces only
            if '"has_error": true' in line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ds in cfg["setup"]["dataset"]:
        setup = ds["content"]
        rows, stats = assign_splits(_stream(Path(setup["path"]), ds["id"]), setup)
        if args.smoke:
            rows = take_smoke(rows, cfg["params"]["smoke_per_split"])
        stats["written"] = len(rows)
        with (args.out / f"{ds['id']}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[ds["id"]] = dict(stats)
        print(ds["id"], dict(stats))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
