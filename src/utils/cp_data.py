"""Load the step-embedding cache written by the cp_step_embeddings experiment.

Inputs
    <emb_dir>/chunk*.npz   see src/experiments/cp_step_embeddings/extract.py
    layer                  the hidden-state index to load (one of the cache's `layers`)

Outputs
    load_cache()   {split name: Split}, each Split holding one float16 array [T_i, d] per trace, the labelled
                   error step y [n], the lengths [n] and the trace ids
    pad_batch()    a padded float32 batch [B, T_max, d] with its boolean mask, on the embeddings' device
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class Split:
    emb: list = field(default_factory=list)       # per trace: [T, d] float16 (numpy, or torch after to_device)
    y: list = field(default_factory=list)
    ids: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def lengths(self) -> np.ndarray:
        return np.array([len(e) for e in self.emb])

    def to_device(self, device) -> "Split":
        return Split([torch.as_tensor(e).to(device) for e in self.emb], list(self.y), list(self.ids))


def available_layers(emb_dir: Path) -> list[int]:
    first = sorted(Path(emb_dir).glob("chunk*.npz"))[0]
    return np.load(first)["layers"].tolist()


def load_cache(emb_dir: Path, layer: int) -> dict[str, Split]:
    splits: dict[str, Split] = {}
    files = sorted(Path(emb_dir).glob("chunk*.npz"))
    if not files:
        raise FileNotFoundError(f"no chunk*.npz in {emb_dir}")
    for path in files:
        z = np.load(path)
        emb, bounds = z[f"emb_L{layer}"], np.concatenate([[0], np.cumsum(z["lengths"])])
        for i, (name, y, tid) in enumerate(zip(z["split"], z["y"], z["ids"])):
            s = splits.setdefault(str(name), Split())
            s.emb.append(emb[bounds[i] : bounds[i + 1]])
            s.y.append(int(y))
            s.ids.append(str(tid))
    return splits


def merge(splits: dict[str, Split], names: list[str]) -> Split:
    out = Split()
    for n in names:
        out.emb += splits[n].emb
        out.y += splits[n].y
        out.ids += splits[n].ids
    return out


def pad_batch(split: Split, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns h [B, T_max, d] float32, mask [B, T_max] bool, y [B] long for the traces in `idx`."""
    embs = [split.emb[i] for i in idx]
    T = max(len(e) for e in embs)
    dev = embs[0].device
    h = torch.zeros(len(embs), T, embs[0].shape[1], dtype=torch.float32, device=dev)
    mask = torch.zeros(len(embs), T, dtype=torch.bool, device=dev)
    for b, e in enumerate(embs):
        h[b, : len(e)] = e.float()
        mask[b, : len(e)] = True
    y = torch.tensor([split.y[i] for i in idx], dtype=torch.long, device=dev)
    return h, mask, y
