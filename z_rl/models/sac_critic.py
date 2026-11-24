# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

"""Action-value heads with a shared observation encoder and Polyak-averaged target heads."""

from __future__ import annotations

import copy

import torch
from torch import nn

from .composition import ComposableModel


class SACCriticModel(ComposableModel):
    def __init__(
        self, *args, num_actions: int, num_qs: int = 2, num_min_qs: int | None = None, init_weights=None, **kwargs
    ) -> None:
        self.num_actions = num_actions
        self.num_qs = num_qs
        self.num_min_qs = num_qs if num_min_qs is None else num_min_qs
        if not 1 <= self.num_min_qs <= num_qs:
            raise ValueError("num_min_qs must be between 1 and num_qs.")
        super().__init__(*args, **kwargs)
        if self.distribution is not None:
            raise ValueError("The SAC critic requires deterministic Q heads.")
        if init_weights is not None:
            for head in self.head:
                head.init_weights(init_weights)
        self.target_heads = copy.deepcopy(self.head).requires_grad_(False)

    def build_head(self, input_dim, output_dim, hidden_dims, activation) -> nn.ModuleList:
        build = super().build_head
        return nn.ModuleList(
            build(input_dim + self.num_actions, output_dim, hidden_dims, activation) for _ in range(self.num_qs)
        )

    def forward(self, obs, actions: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        inputs = torch.cat((self.get_latent(obs), actions), dim=-1)
        heads = self.target_heads if target else self.head
        if target and self.num_min_qs < self.num_qs:
            indices = torch.randperm(self.num_qs, device="cpu")[: self.num_min_qs].tolist()
            return torch.cat([heads[i](inputs) for i in indices], dim=-1)
        return torch.cat([head(inputs) for head in heads], dim=-1)

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        for target, online in zip(self.target_heads.parameters(), self.head.parameters()):
            target.lerp_(online, tau)
