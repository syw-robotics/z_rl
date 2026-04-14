# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from z_rl.modules import MLP
from z_rl.utils import ObsSelector, resolve_obs_temporal_selector

from z_rl.models.composition import ComposableModel, LatentSpec


@dataclass(slots=True)
class MLPEncoderLatentSpec(LatentSpec):
    """Latent spec that encodes the single active ``policy`` observation group with an MLP."""

    latent_dim: int = 128
    encoder_hidden_dims: tuple[int, ...] | list[int] = (256,)
    activation: str = "elu"
    concat_last_obs: bool = False

    def validate(self, model: nn.Module) -> None:
        """Validate that the model exposes exactly one active ``policy`` observation group."""
        if getattr(model, "obs_groups", None) != ["policy"]:
            raise ValueError(
                "`MLPEncoderLatentSpec` requires exactly one active observation group named 'policy'. "
                f"Got {getattr(model, 'obs_groups', None)}."
            )
        if self.latent_dim <= 0:
            raise ValueError(f"`latent_dim` must be positive, got {self.latent_dim}.")
        if isinstance(self.encoder_hidden_dims, (tuple, list)) and len(self.encoder_hidden_dims) == 0:
            raise ValueError("`encoder_hidden_dims` can not be empty.")

        self.last_obs_selector = None
        if self.concat_last_obs:
            self.last_obs_selector = resolve_obs_temporal_selector("policy", "last", model.obs_group_time_slice_map)

    def build_latent_adapter(self, model: nn.Module) -> nn.Module:
        """Build the encoder adapter and cache the optional last-frame selector once."""
        encoder = MLP(model.obs_dim, self.latent_dim, self.encoder_hidden_dims, self.activation)
        return _EncoderLatentAdapter(encoder=encoder, last_obs_selector=self.last_obs_selector)

    def get_latent_dim(self, model: nn.Module) -> int:
        """Return the encoder latent size plus the optional appended last-frame width."""
        latent_dim = self.latent_dim
        if not self.concat_last_obs:
            return latent_dim
        return latent_dim + self.last_obs_selector.dim


class _EncoderLatentAdapter(nn.Module):
    """Latent adapter that encodes the active observation group and optionally appends the last frame."""

    def __init__(self, encoder: nn.Module, last_obs_selector: ObsSelector | None = None) -> None:
        """Store the encoder and the optional cached selector for the last observation frame."""
        super().__init__()
        self.encoder = encoder
        self.last_obs_selector = last_obs_selector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the normalized observation and optionally concatenate the cached last frame."""
        latent = self.encoder(x)
        if self.last_obs_selector is None:
            return latent
        last_obs = self.last_obs_selector.select(x)
        return torch.cat([latent, last_obs], dim=-1)


class EncoderMLPModel(ComposableModel):
    """MLPModel variant whose latent is produced by ``MLPEncoderLatentSpec``.

    Data flow: ``policy obs -> (normalization) -> encoder MLP -> [optional last obs concat] -> head -> (distribution) -> output``.
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
        obs_group_time_slice_map: dict[str, dict[str, ObsSelector]] | None = None,
        latent_dim: int = 128,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256,),
        encoder_activation: str = "elu",
        concat_last_obs: bool = False,
    ) -> None:
        """Initialize the encoder-based MLP model with a latent spec."""
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
            latent_spec=MLPEncoderLatentSpec(
                latent_dim=latent_dim,
                encoder_hidden_dims=encoder_hidden_dims,
                activation=encoder_activation,
                concat_last_obs=concat_last_obs,
            ),
        )
