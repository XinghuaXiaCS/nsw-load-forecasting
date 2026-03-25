from __future__ import annotations

import torch
import torch.nn as nn


class ResidualWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor, baseline: torch.Tensor | None = None) -> torch.Tensor:
        pred = self.backbone(x)
        if baseline is None:
            return pred
        return baseline + pred
