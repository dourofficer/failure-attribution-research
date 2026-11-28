"""Check the embedding cache against the splits and write a manifest.

Inputs
    config.resolved.yaml, <splits>/<dataset_id>.jsonl, <emb>/chunk*.npz

Outputs
    <out>/manifest.json   traces and steps per (dataset, split), layers, embedding width, token-count quantiles
    Exits non-zero if a chunk is missing, ids are out of order, a length or label disagrees with the splits,
    an embedding array has the wrong number of rows, or a value is not finite.

Usage
    uv run python -m src.experiments.cp_step_embeddings.verify --config <resolved> --splits <dir> --emb <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.experiments.cp_step_embeddings.extract import chunks_of, load_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    rows = load_rows(cfg, args.splits)
    traces, steps, n_tokens = Counter(), Counter(), []
    layers = width = None
    for c, rng in enumerate(tqdm(chunks_of(len(rows), cfg["params"]["chunk_size"]), desc="verify", unit="chunk")):
        path = args.emb / f"chunk{c:05d}.npz"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        z = np.load(path)
        chunk = [rows[i] for i in rng]
        if list(z["ids"]) != [r["id"] for r in chunk]:
            raise SystemExit(f"{path}: ids differ from the splits")
        if list(z["lengths"]) != [r["num_steps"] for r in chunk] or list(z["y"]) != [r["y"] for r in chunk]:
            raise SystemExit(f"{path}: lengths or labels differ from the splits")
        if layers is None:
            layers = z["layers"].tolist()
        for idx in layers:
            emb = z[f"emb_L{idx}"]
            if emb.shape[0] != int(z["lengths"].sum()) or not np.isfinite(emb).all():
                raise SystemExit(f"{path}: emb_L{idx} has wrong rows or non-finite values")
            width = emb.shape[1]
        for r in chunk:
            traces[f"{r['dataset']}/{r['split']}"] += 1
            steps[f"{r['dataset']}/{r['split']}"] += r["num_steps"]
        n_tokens.extend(z["n_tokens"].tolist())

    q = np.quantile(n_tokens, [0.5, 0.99, 1.0]).astype(int).tolist()
    manifest = {
        "traces": dict(traces),
        "steps": dict(steps),
        "total_traces": len(rows),
        "layers": layers,
        "width": width,
        "n_tokens_q50_q99_max": q,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
