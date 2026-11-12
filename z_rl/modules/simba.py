from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).clamp_min(1e-8)


class SimBaScale(nn.Module):
    """Learnable per-feature scale used by SimBaV2."""

    def __init__(self, dim: int, init: float, scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), float(scale)))
        self.forward_scale = float(init) / float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self.forward_scale * x


class SimBaLinear(nn.Module):
    """Bias-free linear layer with unit-normalized output rows."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, _normalize(self.weight, dim=1))


class SimBaEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, scale_init: float, scale: float, c_shift: float) -> None:
        super().__init__()
        self.c_shift = float(c_shift)
        self.projection = SimBaLinear(input_dim + 1, hidden_dim)
        self.projection.weight._non_muon = True
        self.scale = SimBaScale(hidden_dim, scale_init, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.full_like(x[..., :1], self.c_shift)
        x = _normalize(torch.cat((x, shift), dim=-1))
        return _normalize(self.scale(self.projection(x)))


class SimBaResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        expansion: int,
        scale_init: float,
        scale: float,
        alpha_init: float,
        alpha_scale: float,
    ) -> None:
        super().__init__()
        expanded_dim = hidden_dim * expansion
        self.up = SimBaLinear(hidden_dim, expanded_dim)
        self.scale = SimBaScale(expanded_dim, scale_init / math.sqrt(expansion), scale / math.sqrt(expansion))
        self.down = SimBaLinear(expanded_dim, hidden_dim)
        self.alpha = SimBaScale(hidden_dim, alpha_init, alpha_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = _normalize(self.down(F.relu(self.scale(self.up(x)))))
        return _normalize(x + self.alpha(residual - x))


class SimBa(nn.Module):
    """SimBaV2 hyperspherical residual network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        expansion: int = 4,
        c_shift: float = 3.0,
    ) -> None:
        super().__init__()
        scale = math.sqrt(2.0 / hidden_dim)
        self.embedder = SimBaEmbedder(input_dim, hidden_dim, scale, scale, c_shift)
        self.blocks = nn.Sequential(
            *(
                SimBaResidualBlock(
                    hidden_dim,
                    expansion,
                    scale,
                    scale,
                    1.0 / (num_blocks + 1),
                    1.0 / math.sqrt(hidden_dim),
                )
                for _ in range(num_blocks)
            )
        )
        self.head = SimBaLinear(hidden_dim, hidden_dim)
        self.head_scale = SimBaScale(hidden_dim, scale, scale)
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.output.bias)
        self.output.weight._non_muon = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.embedder(x))
        return self.output(self.head_scale(self.head(x)))
