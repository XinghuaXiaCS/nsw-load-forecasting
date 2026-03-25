from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class ITransformer(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, pred_len: int, d_model: int, n_heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)
