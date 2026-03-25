from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, seq_len: int, patch_len: int, stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        x = x.transpose(1, 2)  # [B, C, L]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, N, P]
        b, c, n, p = patches.shape
        patches = self.proj(patches.reshape(b * c, n, p))
        return patches, c


class PatchTST(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, pred_len: int, d_model: int, num_heads: int, num_layers: int, patch_len: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.embed = PatchEmbedding(seq_len, patch_len, stride, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches, c = self.embed(x)
        z = self.encoder(patches)
        z = z.mean(dim=1)
        z = z.reshape(x.shape[0], c, -1).mean(dim=1)
        return self.head(z)
