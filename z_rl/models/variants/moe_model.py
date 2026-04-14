# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from tensordict import TensorDict

from z_rl.modules import MoE

from z_rl.models.composition import ComposableModel, HeadSpec


@dataclass(slots=True)
class MoEHeadSpec(HeadSpec):
    """Explicit head spec that builds a Mixture-of-Experts output head."""

    num_experts: int = 4
    expert_hidden_dims: tuple[int, ...] | list[int] | int = (256,)
    gate_hidden_dims: tuple[int, ...] | list[int] | int | None = None

    def validate(self, model: nn.Module) -> None:
        """Validate MoE-specific parameters before the head is rebuilt."""
        if self.num_experts <= 0:
            raise ValueError(f"`num_experts` must be positive, got {self.num_experts}.")
        if isinstance(self.expert_hidden_dims, (tuple, list)) and len(self.expert_hidden_dims) == 0:
            raise ValueError("`expert_hidden_dims` can not be empty.")

    def build_head(self, model: nn.Module, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        """Build the MoE head for the provided model dimensions."""
        return MoE(
            input_dim,
            output_dim,
            self.num_experts,
            self.expert_hidden_dims,
            gate_hidden_dims=self.gate_hidden_dims,
            activation=activation,
        )


class MoEModel(ComposableModel):
    """MLPModel variant whose head is a Mixture-of-Experts MLP.

    Data flow: ``obs groups -> (per-group normalization) -> concat latent -> MoE head -> (distribution) -> output``.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_experts: int = 4,
        expert_hidden_dims: tuple[int, ...] | list[int] | int = (256,),
        gate_hidden_dims: tuple[int, ...] | list[int] | int | None = None,
    ) -> None:
        """Initialize the MoE-based model with an explicit MoE head spec."""
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            head_spec=MoEHeadSpec(
                num_experts=num_experts,
                expert_hidden_dims=expert_hidden_dims,
                gate_hidden_dims=gate_hidden_dims,
            ),
        )
