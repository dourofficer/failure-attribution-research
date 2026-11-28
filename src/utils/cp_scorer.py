"""Step scorer of Work Package 1: a small Transformer encoder over frozen step embeddings.

Inputs
    h      [B, T, d_llm] padded step embeddings
    mask   [B, T] bool, True on real steps

Outputs
    logits [B, T], with -inf on padding, so that softmax over T gives pi_t over the real steps only

Architecture: u_t = LayerNorm(W h_t); a bidirectional pre-norm Transformer encoder (default 2 layers, width 256,
4 heads, feed-forward 1024, dropout 0.1) with rotary position encoding over step positions; a linear head w.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def rotary(x: torch.Tensor) -> torch.Tensor:
    """Apply rotary position encoding to x [B, heads, T, head_dim] (head_dim even), positions 0..T-1."""
    T, dim = x.shape[-2], x.shape[-1]
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * freq[None, :]   # [T, dim/2]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(
            rotary(q), rotary(k), v, attn_mask=mask[:, None, None, :], dropout_p=self.p if self.training else 0.0
        )
        x = x + self.drop(self.proj(attn.transpose(1, 2).reshape(B, T, d)))
        return x + self.drop(self.ff(self.norm2(x)))


class StepScorer(nn.Module):
    def __init__(self, d_llm: int, d: int = 256, layers: int = 2, heads: int = 4, ffn: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(d_llm, d)
        self.inp_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList(EncoderLayer(d, heads, ffn, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.inp_norm(self.inp(h))
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.head(self.out_norm(x)).squeeze(-1)
        return logits.masked_fill(~mask, float("-inf"))
