"""Cache frozen, mean-pooled step embeddings for every trace in the splits (one shard of the work).

Inputs
    config.resolved.yaml          the model setup (path, dtype, depth fractions, formats) and params.chunk_size
    <splits>/<dataset_id>.jsonl   the records written by splits.py

Outputs
    <out>/chunk{c:05d}.npz for every chunk c with c % num_shards == shard, holding
        ids, dataset, split   [n] strings
        lengths, y            [n] ints: steps per trace, first error step
        n_tokens              [n] ints: tokens in the formatted trace
        layers                [L] hidden-state indices kept
        emb_L{idx}            [sum(lengths), d] float16, steps of all traces concatenated in order
    A chunk that already exists is skipped; chunks are written to a temp name and renamed.

Usage
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.experiments.cp_step_embeddings.extract \
        --config <resolved> --splits <dir> --out <dir> --shard 0 --num-shards 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.experiments.cp_step_embeddings.pooling import format_trace, layer_indices, pool_steps, token_step_ids

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_rows(cfg: dict, splits_dir: Path) -> list[dict]:
    """All split records, datasets in the order the config lists them."""
    rows = []
    for ds in cfg["setup"]["dataset"]:
        with (splits_dir / f"{ds['id']}.jsonl").open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f)
    return rows


def chunks_of(n_rows: int, chunk_size: int) -> list[range]:
    return [range(a, min(a + chunk_size, n_rows)) for a in range(0, n_rows, chunk_size)]


@torch.no_grad()
def embed_trace(model, tok, row: dict, m: dict, layers: list[int], captured: dict) -> tuple[dict[int, np.ndarray], int]:
    """One forward pass; returns {layer index: [T, d] float16} and the token count."""
    text, spans = format_trace(row["prompt"], row["steps"], m["prompt_format"], m["step_format"])
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    n_tok = enc["input_ids"].shape[1]
    if n_tok > m["max_tokens"]:
        raise ValueError(f"{row['id']}: {n_tok} tokens > max_tokens={m['max_tokens']}; windowing is not implemented")
    step_ids = torch.from_numpy(token_step_ids(enc["offset_mapping"][0].numpy(), spans)).to(model.device)
    captured.clear()
    out = model(input_ids=enc["input_ids"].to(model.device))
    n_layers = model.config.num_hidden_layers
    embs = {}
    for idx in layers:
        hidden = out.last_hidden_state[0] if idx == n_layers else captured[idx][0]
        pooled = pool_steps(hidden, step_ids, len(spans)).cpu().numpy()
        half = pooled.astype(np.float16)
        if not np.isfinite(half).all():
            raise ValueError(f"{row['id']}: layer {idx} overflows float16")
        embs[idx] = half
    return embs, n_tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    m = cfg["setup"]["model"]["content"]
    rows = load_rows(cfg, args.splits)
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [
        (c, rng)
        for c, rng in enumerate(chunks_of(len(rows), cfg["params"]["chunk_size"]))
        if c % args.num_shards == args.shard and not (args.out / f"chunk{c:05d}.npz").exists()
    ]
    if not todo:
        print(f"shard {args.shard}: nothing to do")
        return

    src = m.get("path") or m["hub_id"]
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(src, torch_dtype=DTYPES[m["dtype"]])
    if m.get("text_submodule"):   # multimodal checkpoint: keep the text decoder only
        model = getattr(model, m["text_submodule"])
    model = model.to("cuda:0").eval()
    n_layers = model.config.num_hidden_layers
    layers = layer_indices(n_layers, m["layer_fractions"])
    captured: dict[int, torch.Tensor] = {}
    for idx in layers:
        if idx != n_layers:   # the final layer is read after the last norm, from the model output
            model.layers[idx - 1].register_forward_hook(
                lambda _mod, _inp, out, idx=idx: captured.__setitem__(idx, out[0] if isinstance(out, tuple) else out)
            )

    bar = tqdm(total=sum(len(r) for _, r in todo), desc=f"shard {args.shard}", unit="trace")
    for c, rng in todo:
        per_layer = {idx: [] for idx in layers}
        n_tokens = []
        for i in rng:
            embs, n_tok = embed_trace(model, tok, rows[i], m, layers, captured)
            for idx in layers:
                per_layer[idx].append(embs[idx])
            n_tokens.append(n_tok)
            bar.update(1)
        chunk = [rows[i] for i in rng]
        tmp = args.out / f".chunk{c:05d}.tmp.npz"
        np.savez(
            tmp,
            ids=np.array([r["id"] for r in chunk]),
            dataset=np.array([r["dataset"] for r in chunk]),
            split=np.array([r["split"] for r in chunk]),
            lengths=np.array([r["num_steps"] for r in chunk], dtype=np.int32),
            y=np.array([r["y"] for r in chunk], dtype=np.int32),
            n_tokens=np.array(n_tokens, dtype=np.int32),
            layers=np.array(layers, dtype=np.int32),
            **{f"emb_L{idx}": np.concatenate(per_layer[idx]) for idx in layers},
        )
        os.replace(tmp, args.out / f"chunk{c:05d}.npz")
    bar.close()


if __name__ == "__main__":
    main()
