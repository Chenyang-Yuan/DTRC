"""Representation, directed distance, flow policy, and residual dynamics networks."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(x: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), half, device=x.device)
    )
    angles = x[:, None] * freqs[None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def mlp_trunk(
    in_dim: int, hidden: int, depth: int, dropout: float = 0.0
) -> nn.Sequential:
    (layers, d) = ([], in_dim)
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden
    return nn.Sequential(*layers)


class RepHead(nn.Module):
    """Map standardized visual features to a control representation."""

    def __init__(
        self, obs_dim: int, rep_dim: int = 128, hidden: int = 512, depth: int = 2
    ):
        super().__init__()
        self.net = nn.Sequential(
            mlp_trunk(obs_dim, hidden, depth), nn.Linear(hidden, rep_dim)
        )
        self.norm = nn.LayerNorm(rep_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(z))


class Quasimetric(nn.Module):
    """Smoothed symmetric distance plus an order-sensitive positive residual.

    The residual is max_j [A_j(goal) - A_j(current)]_+. Reversing arguments
    changes this residual but leaves the symmetric component unchanged.
    The stabilized norm differs slightly from the exact Euclidean metric;
    see docs/method.md for the numerical convention.
    """

    def __init__(
        self, rep_dim: int, sym_dim: int = 64, asym_dim: int = 64, hidden: int = 256
    ):
        super().__init__()
        self.sym = nn.Sequential(
            nn.Linear(rep_dim, hidden), nn.GELU(), nn.Linear(hidden, sym_dim)
        )
        self.asym_dim = int(asym_dim)
        if self.asym_dim > 0:
            self.asym = nn.Sequential(
                nn.Linear(rep_dim, hidden), nn.GELU(), nn.Linear(hidden, asym_dim)
            )
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, e_s: torch.Tensor, e_g: torch.Tensor) -> torch.Tensor:
        sq = (self.sym(e_s) - self.sym(e_g)).pow(2).sum(-1)
        sym = (sq + 1e-08).sqrt() - 0.0001
        if self.asym_dim > 0:
            asym = F.relu(self.asym(e_g) - self.asym(e_s)).max(dim=-1).values
        else:
            asym = torch.zeros_like(sym)
        return (sym + asym) * self.log_scale.exp()


class LatentStep(nn.Module):
    """Action-conditioned residual prediction in control-representation space."""

    def __init__(self, rep_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rep_dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, rep_dim),
        )

    def forward(self, e: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return e + self.net(torch.cat([e, a], dim=-1))


class DynamicsEnsemble(nn.Module):
    """Independent residual predictors in standardized visual-feature space."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 512,
        depth: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            (
                nn.Sequential(
                    mlp_trunk(obs_dim + action_dim, hidden, depth),
                    nn.Linear(hidden, obs_dim),
                )
                for _ in range(heads)
            )
        )

    def all_next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, a], dim=-1)
        return torch.stack([z + head(x) for head in self.heads], dim=0)

    def rollout_loss(
        self, seq_obs: torch.Tensor, seq_actions: torch.Tensor
    ) -> torch.Tensor:
        steps = seq_actions.shape[1]
        z = seq_obs[:, 0].unsqueeze(0).expand(len(self.heads), -1, -1)
        total = 0.0
        for t in range(steps):
            a = seq_actions[:, t].unsqueeze(0).expand(len(self.heads), -1, -1)
            x = torch.cat([z, a], dim=-1)
            z = torch.stack(
                [z[k] + head(x[k]) for (k, head) in enumerate(self.heads)], dim=0
            )
            total = total + F.mse_loss(z, seq_obs[:, t + 1].unsqueeze(0).expand_as(z))
        return total / steps

    @torch.no_grad()
    def rollout(
        self, z: torch.Tensor, chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = chunk.shape[-2]
        preds = z.unsqueeze(0).expand(len(self.heads), *z.shape)
        disagreement = torch.zeros(z.shape[:-1], device=z.device)
        for t in range(k):
            a = (
                chunk[..., t, :]
                .unsqueeze(0)
                .expand(len(self.heads), *chunk.shape[:-2], -1)
            )
            x = torch.cat([preds, a], dim=-1)
            preds = torch.stack(
                [preds[i] + head(x[i]) for (i, head) in enumerate(self.heads)], dim=0
            )
            disagreement = disagreement + preds.std(dim=0).mean(dim=-1)
        return (preds.mean(dim=0), disagreement / k)


class FlowNet(nn.Module):
    """Conditional vector field for Gaussian-to-action flow matching."""

    def __init__(
        self,
        cond_dim: int,
        out_dim: int,
        hidden: int = 512,
        depth: int = 3,
        time_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.trunk = mlp_trunk(cond_dim + out_dim + time_dim, hidden, depth, dropout)
        self.head = nn.Linear(hidden, out_dim)
        self.out_dim = out_dim

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        emb = sinusoidal_embedding(t.reshape(-1), self.time_dim)
        return self.head(self.trunk(torch.cat([cond, x_t, emb], dim=-1)))

    def loss(
        self, x1: torch.Tensor, cond: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
        v = self(x_t, t, cond)
        per = (v - (x1 - x0)).pow(2).mean(dim=-1)
        if weights is not None:
            per = per * weights
        return per.mean()

    @torch.no_grad()
    def score(
        self, x1: torch.Tensor, cond: torch.Tensor, samples: int = 2
    ) -> torch.Tensor:
        total = torch.zeros(x1.shape[0], device=x1.device)
        for _ in range(samples):
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=x1.device) * 0.8 + 0.1
            x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
            total = total + (self(x_t, t, cond) - (x1 - x0)).pow(2).mean(dim=-1)
        return total / samples

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        steps: int = 8,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(
                cond.shape[0], self.out_dim, device=cond.device, generator=generator
            )
        x = noise
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((cond.shape[0],), i * dt, device=cond.device)
            x = x + dt * self(x, t, cond)
        return x


class GoalCondEncoder(nn.Module):
    """Concatenate visual features, horizon embedding, and control features."""

    def __init__(
        self, obs_dim: int, rep_dim: int, horizon_dim: int = 64, use_rep: bool = True
    ):
        super().__init__()
        self.horizon_dim = horizon_dim
        self.use_rep = use_rep
        self.out_dim = 2 * obs_dim + horizon_dim + (2 * rep_dim if use_rep else 0)

    def forward(self, z, z_goal, e, e_goal, h_norm) -> torch.Tensor:
        parts = [z, z_goal, sinusoidal_embedding(h_norm.reshape(-1), self.horizon_dim)]
        if self.use_rep:
            parts += [e, e_goal]
        return torch.cat(parts, dim=-1)
