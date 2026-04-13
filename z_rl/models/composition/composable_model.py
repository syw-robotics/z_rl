# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from tensordict import TensorDict

from ..mlp_model import MLPModel
from .specs import HeadSpec, LatentSpec


class ComposableModel(MLPModel):
    """MLPModel variant that composes optional latent and head specs during base initialization."""

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
        obs_group_time_slice_map: dict[str, dict[str, slice | torch.Tensor]] | None = None,
        latent_spec: LatentSpec | None = None,
        head_spec: HeadSpec | None = None,
    ) -> None:
        """Initialize the base model with explicit latent/head composition."""
        self.latent_spec = latent_spec
        self.head_spec = head_spec
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            obs_group_time_slice_map=obs_group_time_slice_map,
        )

    def build_latent_adapter(self) -> torch.nn.Module:
        """Build the configured latent adapter or fall back to the base identity adapter."""
        if self.latent_spec is None:
            return super().build_latent_adapter()
        self.latent_spec.validate(self)
        return self.latent_spec.build_latent_adapter(self)

    def get_latent_dim(self) -> int:
        """Return the latent dimensionality after optional composition overrides."""
        if self.latent_spec is None:
            return super().get_latent_dim()
        return self.latent_spec.get_latent_dim(self)

    def build_head(
        self,
        input_dim: int,
        output_dim: int | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> torch.nn.Module:
        """Build the configured head or fall back to the default MLP head."""
        if self.head_spec is not None:
            self.head_spec.validate(self)
            return self.head_spec.build_head(self, input_dim, output_dim, activation)
        return super().build_head(input_dim, output_dim, hidden_dims, activation)
