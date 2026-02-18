"""Multi-Scale Shifted Window Transformer (1D) inspired by MSW-Transformer.

This is a lightweight, single-block implementation for sequence inputs where each
document is a 1D token list (e.g., mz_parent-specific tokens). It supports three
window scales with optional shifts, plus a learnable feature fusion across the
scales. The goal is to capture local structure at multiple receptive fields
without quadratic global attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register


def _window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition 1D sequence into non-overlapping windows with padding.

    Args:
        x: tensor of shape (B, L, C).
        window_size: window length.

    Returns:
        windows: (B * num_windows, window_size, C)
        pad: (pad_left, pad_right) applied to length.
    """

    b, l, c = x.shape
    pad_needed = (window_size - l % window_size) % window_size
    pad_left = 0
    pad_right = pad_needed
    if pad_needed:
        x = F.pad(x, (0, 0, 0, pad_right))  # pad length dimension on the right
        l = x.shape[1]
    x = x.view(b, l // window_size, window_size, c)
    windows = x.reshape(-1, window_size, c)
    return windows, (pad_left, pad_right)


def _window_reverse(windows: torch.Tensor, window_size: int, batch: int, length: int, pad: Tuple[int, int]) -> torch.Tensor:
    """Reconstruct sequence from windows and remove padding."""

    pad_left, pad_right = pad
    x = windows.view(batch, -1, window_size, windows.shape[-1])
    x = x.reshape(batch, -1, windows.shape[-1])
    if pad_right:
        x = x[:, : length, :]
    return x


class ShiftedWindowSelfAttention(nn.Module):
    """Shifted window self-attention for 1D sequences."""

    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size or 0
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        b, l, c = x.shape
        if self.shift_size:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)

        windows, pad = _window_partition(x, self.window_size)
        attn_out, _ = self.attn(windows, windows, windows)
        x = _window_reverse(attn_out, self.window_size, b, l, pad)

        if self.shift_size:
            x = torch.roll(x, shifts=self.shift_size, dims=1)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MSWBlock(nn.Module):
    """Single-layer Multi-Scale Shifted Window block with fusion.

    Processes the same input with multiple window attentions and fuses the
    outputs using a learnable softmax weight per scale.
    """

    def __init__(self, dim: int, num_heads: int, window_sizes: Iterable[int], mlp_ratio: float = 4.0, attn_dropout: float = 0.0, drop: float = 0.0):
        super().__init__()
        self.window_sizes: List[int] = list(window_sizes)
        self.attn_branches = nn.ModuleList(
            [
                ShiftedWindowSelfAttention(dim=dim, num_heads=num_heads, window_size=w, shift_size=w // 2, dropout=attn_dropout)
                for w in self.window_sizes
            ]
        )
        self.ln_attn = nn.ModuleList([nn.LayerNorm(dim) for _ in self.window_sizes])
        self.fusion_logits = nn.Parameter(torch.zeros(len(self.window_sizes)))
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)
        self.ln_mlp = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        attn_outputs = []
        for attn, ln in zip(self.attn_branches, self.ln_attn):
            y = attn(x)
            y = ln(y)
            attn_outputs.append(y)

        weights = torch.softmax(self.fusion_logits, dim=0)
        fused = torch.zeros_like(x)
        for w, y in zip(weights, attn_outputs):
            fused = fused + w * y

        # MLP with residual
        z = self.ln_mlp(fused)
        z = self.mlp(z)
        return fused + z


@dataclass
class MSWConfig:
    dim: int = 128
    num_heads: int = 4
    window_sizes: Tuple[int, int, int] = (5, 10, 20)
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.0
    dropout: float = 0.0


class MSWTransformer(nn.Module):
    """Single-block MSW Transformer for 1D documents."""

    def __init__(self, config: MSWConfig):
        super().__init__()
        self.config = config
        self.block = MSWBlock(
            dim=config.dim,
            num_heads=config.num_heads,
            window_sizes=config.window_sizes,
            mlp_ratio=config.mlp_ratio,
            attn_dropout=config.attn_dropout,
            drop=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C=config.dim)
        return self.block(x)


@register("msw_transformer")
def build_msw_transformer(**kwargs):
    config = MSWConfig(**kwargs)
    return MSWTransformer(config)
