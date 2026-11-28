"""Normalize step-level LRM / PRM datasets under data/lrm_traces/original/ into a unified JSONL schema.

Outputs:
    data/lrm_traces/normalized/{dataset}/data.jsonl
    data/lrm_traces/normalized/stats.json

Usage:
    uv run python -m src.utils.normalize_lrm_traces
    uv run python -m src.utils.normalize_lrm_traces --datasets grace processbench
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import datasets as hf_datasets
import pyarrow.parquet as pq

csv.field_size_limit(sys.maxsize)

ALL_DATASETS = (
    "grace",
    "processbench",
    "prmbench",
    "math_shepherd",
    "versaprm",
    "actprm",
    "genprm",
)

_BOXED_JUDGE_RE = re.compile(r"\\boxed\{(Yes|No)\}")


def stable_task_id(prefix: str, text: str) -> str:
    """Deterministic 16-hex task grouping key from normalized prompt text."""
    norm = " ".join(text.split()).strip()
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def build_record(
    *,
    record_id: str,
    dataset: str,
    subset: str,
    split: str,
    task_id: str,
    prompt: str,
    steps: list[str],
    step_labels: list[int | None],
    label_granularity: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct and validate a single normalized attribution record."""
    if not steps:
        raise ValueError(f"{record_id}: empty steps")
    if len(steps) != len(step_labels):
        raise ValueError(f"{record_id}: len(steps)={len(steps)} != len(step_labels)={len(step_labels)}")
    if label_granularity not in ("all_steps", "first_error_only"):
        raise ValueError(f"{record_id}: invalid label_granularity={label_granularity}")

    clean_labels: list[int | None] = []
    for v in step_labels:
        if v is None:
            if label_granularity == "all_steps":
                raise ValueError(f"{record_id}: None label in all_steps dataset")
            clean_labels.append(None)
        elif v in (0, 1):
            clean_labels.append(int(v))
        else:
            raise ValueError(f"{record_id}: invalid step label {v!r}")

    error_steps = [i for i, v in enumerate(clean_labels) if v == 0]
    first_error = error_steps[0] if error_steps else None

    if label_granularity == "first_error_only" and first_error is not None:
        if len(error_steps) != 1:
            raise ValueError(f"{record_id}: first_error_only record has multiple 0s: {error_steps}")
        if any(v != 1 for v in clean_labels[:first_error]):
            raise ValueError(f"{record_id}: non-1 before first_error")
        if any(v is not None for v in clean_labels[first_error + 1 :]):
            raise ValueError(f"{record_id}: non-None after first_error")

    return {
        "id": record_id,
        "dataset": dataset,
        "subset": subset,
        "split": split,
        "task_id": task_id,
        "prompt": prompt,
        "steps": steps,
        "num_steps": len(steps),
        "step_labels": clean_labels,
        "error_steps": error_steps,
        "first_error_step": first_error,
        "has_error": bool(error_steps),
        "label_granularity": label_granularity,
        "metadata": metadata or {},
    }


def _format_grace_prompt(row: dict[str, Any]) -> str:
    parts: list[str] = []
    ctx = (row.get("context") or "").strip()
    if ctx:
        parts.append(f"Context:\n{ctx}")
    q = (row.get("question") or "").strip()
    if q:
        parts.append(f"Question:\n{q}")
    opts = row.get("options")
    if opts:
        opt_lines = "\n".join(str(o) for o in opts)
        parts.append(f"Options:\n{opt_lines}")
    return "\n\n".join(parts)


def iter_grace(orig_dir: Path, skipped: Counter) -> Iterator[dict[str, Any]]:
    root = orig_dir / "grace" / "dataset"
    for split in ("train", "test"):
        for path in sorted((root / split).glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    raw_steps = row.get("steps") or []
                    if not raw_steps:
                        skipped["empty_steps"] += 1
                        continue
                    steps = [str(s["text"]) for s in raw_steps]
                    labels = [1 if s["faithfulness"] == "faithful" else 0 for s in raw_steps]
                    subset = str(row.get("dataset") or path.stem)
                    orig_qid = str(row.get("original_question_id") or "")
                    task_id = f"grace:{orig_qid}" if orig_qid else stable_task_id("grace", _format_grace_prompt(row))
                    err_cats = [s.get("error_category") for s in raw_steps if s["faithfulness"] != "faithful"]
                    yield build_record(
                        record_id=f"grace:{split}:{row['id']}",
                        dataset="grace",
                        subset=subset,
                        split=split,
                        task_id=task_id,
                        prompt=_format_grace_prompt(row),
                        steps=steps,
                        step_labels=labels,
                        label_granularity="all_steps",
                        metadata={
                            "track": row.get("track"),
                            "model": row.get("model"),
                            "gold_answer": row.get("gold_answer"),
                            "final_answer": row.get("final_answer"),
                            "error_categories": err_cats,
                        },
                    )


def iter_processbench(orig_dir: Path, skipped: Counter) -> Iterator[dict[str, Any]]:
    root = orig_dir / "processbench"
    for path in sorted(root.glob("*.json")):
        subset = path.stem
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            steps = [str(s) for s in (row.get("steps") or [])]
            if not steps:
                skipped["empty_steps"] += 1
                continue
            lbl = int(row["label"])
            if lbl < -1 or lbl >= len(steps):
                skipped["oob_label"] += 1
                continue
            if lbl == -1:
                step_labels: list[int | None] = [1] * len(steps)
            else:
                step_labels = [1] * lbl + [0] + [None] * (len(steps) - lbl - 1)
            prompt = str(row["problem"])
            yield build_record(
                record_id=f"processbench:{row['id']}",
                dataset="processbench",
                subset=subset,
                split="test",
                task_id=stable_task_id(f"processbench:{subset}", prompt),
                prompt=prompt,
                steps=steps,
                step_labels=step_labels,
                label_granularity="first_error_only",
                metadata={
                    "generator": row.get("generator"),
                    "final_answer_correct": row.get("final_answer_correct"),
                },
            )


def iter_prmbench(orig_dir: Path, skipped: Counter) -> Iterator[dict[str, Any]]:
    path = orig_dir / "prmbench" / "prmbench_preview.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            steps = [str(s) for s in (row.get("modified_process") or [])]
            if not steps:
                skipped["empty_steps"] += 1
                continue
            n = len(steps)
            raw_errs = row.get("error_steps") or []
            # Upstream error_steps are 1-indexed; filter out occasional hallucinated indices > T.
            valid_0based = sorted({int(e) - 1 for e in raw_errs if 1 <= int(e) <= n})
            had_oob = any(int(e) < 1 or int(e) > n for e in raw_errs)
            if raw_errs and not valid_0based:
                skipped["all_errors_oob"] += 1
                continue
            err_set = set(valid_0based)
            labels = [0 if idx in err_set else 1 for idx in range(n)]
            prompt = str(row.get("modified_question") or row.get("question") or "")
            subset = str(row.get("classification") or "default")
            orig_id = str(row.get("idx") or i)
            yield build_record(
                record_id=f"prmbench:{orig_id}",
                dataset="prmbench",
                subset=subset,
                split="test",
                task_id=stable_task_id("prmbench", str(row.get("original_question") or prompt)),
                prompt=prompt,
                steps=steps,
                step_labels=labels,
                label_granularity="all_steps",
                metadata={
                    "classification": subset,
                    "had_oob_error_steps": had_oob,
                    "reason": row.get("reason"),
                },
            )


def iter_math_shepherd(orig_dir: Path, skipped: Counter) -> Iterator[dict[str, Any]]:
    root = orig_dir / "math_shepherd" / "data"
    for split, fname in (("train", "train-00000-of-00001.parquet"), ("test", "test-00000-of-00001.parquet")):
        pf = pq.ParquetFile(root / fname)
        idx = 0
        for batch in pf.iter_batches(batch_size=8192):
            prompts = batch.column("prompt").to_pylist()
            completions = batch.column("completions").to_pylist()
            labels_col = batch.column("labels").to_pylist()
            for prompt, steps, raw_lbls in zip(prompts, completions, labels_col):
                idx += 1
                if not steps or len(steps) != len(raw_lbls):
                    skipped["invalid_len"] += 1
                    continue
                step_labels = [1 if bool(v) else 0 for v in raw_lbls]
                yield build_record(
                    record_id=f"math_shepherd:{split}:{idx - 1}",
                    dataset="math_shepherd",
                    subset="default",
                    split=split,
                    task_id=stable_task_id("math_shepherd", str(prompt)),
                    prompt=str(prompt),
                    steps=[str(s) for s in steps],
                    step_labels=step_labels,
                    label_granularity="all_steps",
                )


def iter_versaprm(orig_dir: Path, skipped: Counter) -> Iterator[dict[str, Any]]:
    path = orig_dir / "versaprm_train" / "train" / "file.csv"
    with path.open("r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            try:
                steps = [str(s) for s in ast.literal_eval(row["chain_of_thoughts"])]
                raw_lbls = ast.literal_eval(row["labels"])
            except Exception:
                skipped["parse_error"] += 1
                continue
            if not steps or len(steps) != len(raw_lbls):
                skipped["invalid_len"] += 1
                continue
            step_labels = [1 if int(v) == 1 else 0 for v in raw_lbls]
            qid = str(row.get("id") or "")
            prompt = str(row["question"])
            subset = str(row.get("category") or "default")
            yield build_record(
                record_id=f"versaprm:{qid}:{row.get('cot_id', i)}",
                dataset="versaprm",
                subset=subset,
                split="train",
                task_id=f"versaprm:{qid}" if qid else stable_task_id("versaprm", prompt),
                prompt=prompt,
                steps=steps,
                step_labels=step_labels,
                label_granularity="all_steps",
                metadata={
                    "src": row.get("src"),
                    "gold_answer": row.get("answer"),
                    "parsed_answer": row.get("parsed_answer"),
                    "final_answer_correct": row.get("parsed_answer_correctness") == "True",
                },
            )


def iter_actprm(orig_dir: Path, skipped: Counter) -> Iterator[dict[str, Any]]:
    ds = hf_datasets.load_from_disk(str(orig_dir / "actprm"))
    for i, row in enumerate(ds):
        steps = [str(s) for s in (row["steps"] or [])]
        hl = row["hard_labels"] or []
        if not steps or not hl or len(hl) > len(steps):
            skipped["invalid_len"] += 1
            continue
        zeros = [k for k, v in enumerate(hl) if int(v) == 0]
        if len(zeros) > 1:
            skipped["multi_zero_unexpected"] += 1
            continue
        if zeros:
            first_err = zeros[0]
            if first_err != len(hl) - 1:
                skipped["zero_not_prefix_end"] += 1
                continue
            step_labels: list[int | None] = [1] * first_err + [0] + [None] * (len(steps) - first_err - 1)
        else:
            if len(hl) != len(steps):
                skipped["incomplete_positive_prefix"] += 1
                continue
            step_labels = [1] * len(steps)
        prompt = str(row["question"])
        yield build_record(
            record_id=f"actprm:train:{i}",
            dataset="actprm",
            subset="math",
            split="train",
            task_id=stable_task_id("actprm", prompt),
            prompt=prompt,
            steps=steps,
            step_labels=step_labels,
            label_granularity="first_error_only",
            metadata={
                "orig_id": row["id"],
                "gold_answer": row.get("answer"),
            },
        )


def iter_genprm(orig_dir: Path, skipped: Counter) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(orig_dir / "genprm" / "data" / "train-00000-of-00001.parquet")
    idx = 0
    for batch in pf.iter_batches(batch_size=2048):
        for conv in batch.column("conversations").to_pylist():
            idx += 1
            users = [m["content"] for m in conv if m["role"] == "user"]
            assts = [m["content"] for m in conv if m["role"] == "assistant"]
            if not users or len(users) != len(assts):
                skipped["turn_mismatch"] += 1
                continue
            u0 = users[0]
            if not u0.startswith("Question:") or "\n\n" not in u0:
                skipped["bad_first_turn"] += 1
                continue
            prompt_part, step0 = u0[len("Question:") :].split("\n\n", 1)
            prompt = prompt_part.strip()
            steps = [step0.strip()] + [str(u).strip() for u in users[1:]]
            labels: list[int | None] = []
            bad_judge = False
            for a in assts:
                m = _BOXED_JUDGE_RE.search(a)
                if not m:
                    bad_judge = True
                    break
                labels.append(1 if m.group(1) == "Yes" else 0)
            if bad_judge or any(not s for s in steps):
                skipped["bad_judgement_or_empty_step"] += 1
                continue
            yield build_record(
                record_id=f"genprm:train:{idx - 1}",
                dataset="genprm",
                subset="math",
                split="train",
                task_id=stable_task_id("genprm", prompt),
                prompt=prompt,
                steps=steps,
                step_labels=labels,
                label_granularity="all_steps",
            )


ITERATORS = {
    "grace": iter_grace,
    "processbench": iter_processbench,
    "prmbench": iter_prmbench,
    "math_shepherd": iter_math_shepherd,
    "versaprm": iter_versaprm,
    "actprm": iter_actprm,
    "genprm": iter_genprm,
}


def _quantile(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = int(q * (len(sorted_vals) - 1))
    return float(sorted_vals[pos])


def normalize_dataset(name: str, orig_dir: Path, norm_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    out_path = norm_dir / name / "data.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped: Counter = Counter()

    total = 0
    with_err = 0
    multi_err = 0
    last_only_err = 0
    step_lens: list[int] = []
    err_counts: list[int] = []
    splits: Counter = Counter()
    subsets: Counter = Counter()
    task_ids: set[str] = set()
    granularity = None

    with out_path.open("w", encoding="utf-8") as f:
        for rec in ITERATORS[name](orig_dir, skipped):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            T = rec["num_steps"]
            step_lens.append(T)
            splits[rec["split"]] += 1
            subsets[rec["subset"]] += 1
            task_ids.add(rec["task_id"])
            granularity = rec["label_granularity"]
            if rec["has_error"]:
                with_err += 1
                n_err = len(rec["error_steps"])
                err_counts.append(n_err)
                if n_err > 1:
                    multi_err += 1
                if rec["error_steps"] == [T - 1]:
                    last_only_err += 1

    step_lens.sort()
    size_mb = out_path.stat().st_size / (1024 * 1024)
    stats = {
        "dataset": name,
        "path": str(out_path),
        "size_mb": round(size_mb, 2),
        "label_granularity": granularity,
        "total_traces": total,
        "failed_traces": with_err,
        "failed_pct": round(100.0 * with_err / total, 1) if total else 0.0,
        "multi_error_traces": multi_err,
        "last_step_only_pct_of_failed": round(100.0 * last_only_err / with_err, 1) if with_err else 0.0,
        "unique_tasks": len(task_ids),
        "steps_min": step_lens[0] if step_lens else 0,
        "steps_median": _quantile(step_lens, 0.5),
        "steps_mean": round(sum(step_lens) / len(step_lens), 2) if step_lens else 0.0,
        "steps_p90": _quantile(step_lens, 0.9),
        "steps_max": step_lens[-1] if step_lens else 0,
        "mean_error_steps_when_failed": round(sum(err_counts) / len(err_counts), 2) if err_counts else 0.0,
        "splits": dict(splits),
        "num_subsets": len(subsets),
        "skipped": dict(skipped),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(
        f"[{name:14s}] total={total:7d} | failed={with_err:7d} ({stats['failed_pct']:4.1f}%) | "
        f"tasks={len(task_ids):6d} | T(med/mean/p90)={stats['steps_median']:.0f}/{stats['steps_mean']:.1f}/{stats['steps_p90']:.0f} | "
        f"last_only={stats['last_step_only_pct_of_failed']:4.1f}% | skipped={dict(skipped)} | {size_mb:.1f} MB ({stats['elapsed_s']}s)",
        flush=True,
    )
    return stats


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orig", default="data/lrm_traces/original")
    ap.add_argument("--out", default="data/lrm_traces/normalized")
    ap.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=list(ALL_DATASETS))
    args = ap.parse_args(list(argv) if argv is not None else None)

    orig_dir = Path(args.orig)
    norm_dir = Path(args.out)
    norm_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for name in args.datasets:
        all_stats[name] = normalize_dataset(name, orig_dir, norm_dir)

    stats_path = norm_dir / "stats.json"
    if stats_path.exists():
        prev = json.loads(stats_path.read_text(encoding="utf-8"))
        prev.update(all_stats)
        all_stats = prev
    stats_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote stats to {stats_path}")


if __name__ == "__main__":
    main()
