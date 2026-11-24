# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Derived from FlashSAC, Copyright (c) 2026 Holiday Robotics, under the MIT license.

from __future__ import annotations

import copy
import math
import torch
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn.functional import normalize, softplus
from typing import ClassVar

from z_rl.models.composition.adapters import ObsLatentAdapter


class UnitLinear(nn.Module):
    """Batched linear projection with unit-length weight rows."""

    def __init__(self, ensemble_size: int, input_dim: int, output_dim: int, *, bias: bool = False) -> None:
        """Initialize the orthogonal projections for the ensemble."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, output_dim, input_dim))
        for weight in self.weight.detach():
            nn.init.orthogonal_(weight)
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, output_dim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        """Project the input with its corresponding ensemble weights."""
        result = x @ self.weight.transpose(1, 2)
        return result if self.bias is None else result + self.bias

    def normalize_parameters(self) -> None:
        """Project the weight rows back onto the unit sphere."""
        self.weight.copy_(normalize(self.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    """Independent BatchNorm statistics for each member of the ensemble."""

    def __init__(self, ensemble_size: int, dim: int) -> None:
        """Initialize affine parameters and running moments."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ensemble_size, 1, dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, dim))
        self.register_buffer("running_mean", torch.zeros_like(self.bias))
        self.register_buffer("running_var", torch.ones_like(self.bias))

    def forward(self, x: Tensor, training: bool) -> Tensor:
        """Normalize with batch moments in training and running moments during inference."""
        mean, var = self.running_mean, self.running_var
        if training:
            var, mean = torch.var_mean(x, dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                self.running_mean.lerp_(mean, 0.01)
                self.running_var.lerp_(var * (x.shape[1] / (x.shape[1] - 1)), 0.01)
        return (x - mean) * torch.rsqrt(var + 1e-5) * self.weight + self.bias

    def normalize_parameters(self) -> None:
        """Normalize the combined length of the affine weight and bias."""
        length = (self.weight.square() + self.bias.square()).mean(-1, keepdim=True).add(1e-8).rsqrt()
        self.weight.mul_(length)
        self.bias.mul_(length)


class FlashSACEncoder(nn.Module):
    """Residual encoder used by both the policy and the twin critics."""

    def __init__(self, ensemble_size: int, input_dim: int, hidden_dim: int) -> None:
        """Initialize the embedding and two residual blocks."""
        super().__init__()
        self.input_norm = UnitBatchNorm(ensemble_size, input_dim)
        self.input_proj = UnitLinear(ensemble_size, input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.ModuleList((
                UnitLinear(ensemble_size, hidden_dim, 4 * hidden_dim),
                UnitBatchNorm(ensemble_size, 4 * hidden_dim),
                UnitLinear(ensemble_size, 4 * hidden_dim, hidden_dim),
                UnitBatchNorm(ensemble_size, hidden_dim),
            ))
            for _ in range(2)
        )
        self.output_scale = nn.Parameter(torch.ones(ensemble_size, 1, hidden_dim))

    def forward(self, obs: Tensor, training: bool) -> Tensor:
        """Encode flat observations for every ensemble member."""
        x = obs.unsqueeze(0).expand(self.output_scale.shape[0], -1, -1)
        x = self.input_proj(self.input_norm(x, training))
        for expand, expand_norm, project, project_norm in self.blocks:
            hidden = expand_norm(expand(x), training).relu()
            x = x + project_norm(project(hidden), training).relu()
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.output_scale

    def normalize_parameters(self) -> None:
        """Normalize the affine RMS scale after an optimizer step."""
        norm = self.output_scale.square().mean(-1, keepdim=True).add(1e-8).rsqrt()
        self.output_scale.mul_(norm)


class FlashSACActor(nn.Module):
    """Squashed Gaussian policy with a deterministic VecEnv inference interface."""

    is_recurrent = False

    def __init__(self, obs_groups: list[str], obs_dim: int, hidden_dim: int, low: Tensor, high: Tensor) -> None:
        """Build the policy and record the action space transform."""
        super().__init__()
        self.observations = ObsLatentAdapter(obs_groups, nn.Identity())
        self.obs_dim = obs_dim
        self.encoder = FlashSACEncoder(1, obs_dim, hidden_dim)
        self.mean = UnitLinear(1, hidden_dim, low.numel(), bias=True)
        self.log_std = UnitLinear(1, hidden_dim, low.numel(), bias=True)
        self.register_buffer("action_center", (high + low) / 2)
        self.register_buffer("action_scale", (high - low) / 2)
        self.register_buffer("output_std", torch.zeros_like(low), persistent=False)

    def distribution(self, obs: Tensor, *, training: bool) -> tuple[Tensor, Tensor]:
        """Return the Gaussian parameters before applying tanh."""
        latent = self.encoder(obs, training)
        mean = self.mean(latent).squeeze(0)
        log_std = -4.0 + 6.0 * self.log_std(latent).squeeze(0).tanh()
        return mean, log_std

    def sample(self, obs: Tensor, *, training: bool) -> tuple[Tensor, Tensor]:
        """Draw reparameterized actions and return their corrected log probabilities."""
        mean, log_std = self.distribution(obs, training=training)
        noise = torch.randn_like(mean)
        raw = mean + log_std.exp() * noise
        log_prob = -0.5 * noise.square() - log_std - 0.5 * math.log(2 * math.pi)
        log_det = 2 * (math.log(2) - raw - softplus(-2 * raw))
        return raw.tanh(), (log_prob - log_det).sum(-1)

    def to_env_actions(self, actions: Tensor) -> Tensor:
        """Map unit-range actions to VecEnv action units."""
        return self.action_center + self.action_scale * actions

    def forward(self, obs: TensorDict) -> Tensor:
        """Run deterministic inference on grouped observations."""
        mean, _ = self.distribution(self.observations(obs), training=False)
        return self.to_env_actions(mean.tanh())

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return the deterministic policy with a flat input for ONNX export."""
        return _OnnxFlashSACActor(copy.deepcopy(self))


class _OnnxFlashSACActor(nn.Module):
    input_names: ClassVar[list[str]] = ["obs"]
    output_names: ClassVar[list[str]] = ["actions"]

    def __init__(self, actor: FlashSACActor) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, obs: Tensor) -> Tensor:
        mean, _ = self.actor.distribution(obs, training=False)
        return self.actor.to_env_actions(mean.tanh())

    def get_dummy_inputs(self) -> tuple[Tensor]:
        return (self.actor.action_center.new_zeros(1, self.actor.obs_dim),)


class FlashSACCritic(nn.Module):
    """Two categorical action-value models evaluated in parallel."""

    def __init__(self, obs_dim: int, hidden_dim: int, num_actions: int, support: Tensor) -> None:
        """Build the twin critics over the given bin support."""
        super().__init__()
        self.encoder = FlashSACEncoder(2, obs_dim + num_actions, hidden_dim)
        self.head = UnitLinear(2, hidden_dim, support.numel(), bias=True)
        self.register_buffer("support", support)

    def forward(self, obs: Tensor, action: Tensor, *, training: bool) -> tuple[Tensor, Tensor]:
        """Return each critic's expected value and categorical log probabilities."""
        latent = self.encoder(torch.cat((obs, action), dim=-1), training)
        log_probs = self.head(latent).log_softmax(dim=-1)
        return (log_probs.exp() * self.support).sum(-1), log_probs


@torch.no_grad()
def normalize_parameters(network: nn.Module) -> None:
    """Apply the FlashSAC weight constraints to an actor or critic."""
    for module in network.modules():
        if isinstance(module, (UnitLinear, UnitBatchNorm, FlashSACEncoder)):
            module.normalize_parameters()
