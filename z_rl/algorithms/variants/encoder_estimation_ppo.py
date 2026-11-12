# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch.nn.functional as F

from z_rl.storage import RolloutStorage
from z_rl.utils import resolve_target_obs_term_selector

from ..composition import ComposablePPO, PPOLossSpec


@dataclass
class EncoderEstimationLossSpec(PPOLossSpec):
    """Fit a leading actor latent slice to selected critic observation terms.

    Estimation index rule: the prediction is always ``latent[..., 0:target_dim]``.
    """

    target_obs_group_name: str = "critic"
    target_obs_term_names: Sequence[str] = ("base_lin_vel",)
    estimation_loss_coef: float = 1.0

    def validate(self, algo: object) -> None:
        """Resolve the target observation slice from actor observation metadata."""
        actor = getattr(algo, "actor", None)
        if not callable(getattr(actor, "get_latent", None)):
            raise ValueError(
                f"`EncoderEstimationLossSpec` requires `algo.actor` to expose `get_latent()`, got {type(actor)}."
            )
        obs_format = getattr(actor, "obs_format", None)
        time_slice_map = getattr(actor, "obs_group_time_slice_map", None)
        if not obs_format or not time_slice_map:
            raise ValueError(
                "`EncoderEstimationLossSpec` requires the actor to expose `obs_format` and `obs_group_time_slice_map`."
            )

        self.target_obs_selector = resolve_target_obs_term_selector(
            target_obs_group_name=self.target_obs_group_name,
            target_obs_term_names=self.target_obs_term_names,
            obs_group_time_slice_map=time_slice_map,
            obs_format=obs_format,
        )

        actor_latent_dim = actor.latent_dim
        if actor_latent_dim < self.target_obs_selector.dim:
            raise ValueError(
                "`EncoderEstimationLossSpec` can not infer an estimation latent slice because the actor latent is "
                f"smaller than target_dim={self.target_obs_selector.dim}: actor_latent_dim={actor_latent_dim}."
            )
        print("[EncoderEstimationLossSpec] Resolved estimation loss target_obs_selector:", self.target_obs_selector)

    def compute(self, algo: object, minibatch: RolloutStorage.Batch):
        """MSE between the selected actor latent slice and the critic observation target."""
        obs = minibatch.observations
        target = self.target_obs_selector.select(obs[self.target_obs_group_name])
        latent = algo.actor_forward_context["latent"]  # type: ignore[attr-defined]
        prediction = latent[..., : target.shape[-1]]
        return {"estimation_loss": F.mse_loss(prediction, target)}, {}


class EncoderEstimationPPO(ComposablePPO):
    """Named PPO preset that installs ``EncoderEstimationLossSpec``."""

    loss_spec_class = EncoderEstimationLossSpec
