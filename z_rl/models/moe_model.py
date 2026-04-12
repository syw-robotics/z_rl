# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from tensordict import TensorDict

from z_rl.models.mixins.head_mixin import MoEHeadMixin
from z_rl.models.mlp_model import MLPModel


class MoEModel(MoEHeadMixin, MLPModel):
    """MLPModel variant whose head is a Mixture-of-Experts MLP.

    Data flow: ``obs groups -> (per-group normalization) -> concat latent -> MoE head -> (distribution) -> output``.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_experts: int = 4,
        expert_hidden_dims: tuple[int, ...] | list[int] | int = (256,),
        gate_hidden_dims: tuple[int, ...] | list[int] | int | None = None,
    ) -> None:
        """Initialize the MoE-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Unused for MoE head, kept for API compatibility with MLPModel.
            activation: Activation function of expert MLPs.
            obs_normalization: Whether to normalize the observations before feeding them to the model.
            distribution_cfg: Configuration dictionary for the output distribution.
            num_experts: Number of MoE experts.
            expert_hidden_dims: Hidden dimensions of each expert MLP.
            gate_hidden_dims: Hidden dimensions of the gate MLP. If ``None``, use a single linear layer as gate.
        """
        # Keep signature compatible with MLPModel while replacing the post-latent MLP head with MoE.
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        self.num_experts = num_experts
        self.expert_hidden_dims = expert_hidden_dims
        self.gate_hidden_dims = gate_hidden_dims
        self.init_moe_head(output_dim, activation)
