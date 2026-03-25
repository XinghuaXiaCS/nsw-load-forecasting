from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0, device=t.device)) / max(half - 1, 1)))
        angles = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ConditionalDiffusionNet(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.time_embed = TimeEmbedding(hidden_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.x_proj = nn.Conv1d(1, hidden_dim, kernel_size=1)
        self.block1 = ConvBlock(hidden_dim)
        self.block2 = ConvBlock(hidden_dim)
        self.out = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, noisy_future: torch.Tensor, t: torch.Tensor, context_summary: torch.Tensor) -> torch.Tensor:
        # noisy_future [B, H], context_summary [B, D]
        h = self.x_proj(noisy_future.unsqueeze(1))
        te = self.time_embed(t.float()).unsqueeze(-1)
        ce = self.context_encoder(context_summary).unsqueeze(-1)
        h = h + te + ce
        h = self.block1(h)
        h = self.block2(h)
        return self.out(h).squeeze(1)


class DiffusionForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int, timesteps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        self.horizon = horizon
        self.timesteps = timesteps
        self.net = ConditionalDiffusionNet(input_dim=input_dim, horizon=horizon, hidden_dim=hidden_dim)
        beta = torch.linspace(beta_start, beta_end, timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def summarize_context(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bar[t].unsqueeze(-1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    def loss(self, x_context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = target.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=target.device)
        noise = torch.randn_like(target)
        noisy = self.q_sample(target, t, noise)
        context_summary = self.summarize_context(x_context)
        pred_noise = self.net(noisy, t, context_summary)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, x_context: torch.Tensor, n_samples: int = 50) -> torch.Tensor:
        batch = x_context.shape[0]
        context_summary = self.summarize_context(x_context)
        draws = []
        for _ in range(n_samples):
            x = torch.randn(batch, self.horizon, device=x_context.device)
            for step in reversed(range(self.timesteps)):
                t = torch.full((batch,), step, device=x_context.device, dtype=torch.long)
                pred_noise = self.net(x, t, context_summary)
                alpha = self.alpha[step]
                alpha_bar = self.alpha_bar[step]
                x = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
                if step > 0:
                    x = x + torch.sqrt(self.beta[step]) * torch.randn_like(x)
            draws.append(x.unsqueeze(1))
        return torch.cat(draws, dim=1)
