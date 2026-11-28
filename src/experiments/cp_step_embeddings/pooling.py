"""Turn a trace into text with per-step character spans, and pool token hidden states per step.

Inputs
    a prompt, its ordered step texts, and the two format strings of the model setup
    token character offsets from a fast tokenizer, and hidden states [n_tokens, d]

Outputs
    format_trace()     the trace text and one (start, end) character span per step, covering the step text only
    token_step_ids()   for every token, the step it belongs to, or -1
    pool_steps()       mean hidden state per step, [T, d]
    layer_indices()    which hidden-state index each depth fraction maps to
"""

from __future__ import annotations

import numpy as np
import torch


def format_trace(prompt: str, steps: list[str], prompt_format: str, step_format: str) -> tuple[str, list[tuple[int, int]]]:
    parts = [prompt_format.format(prompt=prompt)]
    pos = len(parts[0])
    spans = []
    for k, step in enumerate(steps, start=1):
        head, tail = step_format.split("{step}")
        head = head.format(k=k)
        spans.append((pos + len(head), pos + len(head) + len(step)))
        block = head + step + tail.format(k=k)
        parts.append(block)
        pos += len(block)
    return "".join(parts), spans


def token_step_ids(offsets: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Assign each token to the step whose span contains the token's first character; -1 otherwise.

    `offsets` is [n_tokens, 2] (start, end). Zero-width tokens (start == end) are never assigned.
    """
    starts = np.asarray(offsets)[:, 0]
    width = np.asarray(offsets)[:, 1] - starts
    span_lo = np.array([a for a, _ in spans])
    span_hi = np.array([b for _, b in spans])
    idx = np.searchsorted(span_lo, starts, side="right") - 1   # last span starting at or before the token
    ok = (idx >= 0) & (starts < span_hi[np.clip(idx, 0, None)]) & (width > 0)
    return np.where(ok, idx, -1)


def pool_steps(hidden: torch.Tensor, step_ids: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Mean of `hidden` [n_tokens, d] over the tokens of each step. Raises if a step has no token."""
    mask = step_ids >= 0
    counts = torch.bincount(step_ids[mask], minlength=n_steps)
    if (counts == 0).any():
        raise ValueError(f"steps without tokens: {(counts == 0).nonzero().flatten().tolist()}")
    out = torch.zeros(n_steps, hidden.shape[1], dtype=torch.float32, device=hidden.device)
    out.index_add_(0, step_ids[mask], hidden[mask].float())
    return out / counts[:, None]


def layer_indices(n_layers: int, fractions: list[float]) -> list[int]:
    """Hidden-state index (1 = output of the first block, n_layers = output of the last) per depth fraction."""
    idx = [max(1, round(f * n_layers)) for f in fractions]
    if len(set(idx)) != len(idx):
        raise ValueError(f"depth fractions {fractions} collide on {n_layers} layers: {idx}")
    return idx
