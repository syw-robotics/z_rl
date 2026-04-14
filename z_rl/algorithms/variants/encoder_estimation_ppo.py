# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from z_rl.env import VecEnv
from z_rl.models import EncoderMLPModel
from z_rl.storage import RolloutStorage
from z_rl.utils import ObsSelector, resolve_target_obs_term_selector

from ..composition import ComposablePPO, PPOLossSpec


@dataclass(slots=True)
class EncoderEstimationLossSpec(PPOLossSpec):
    """PPO loss spec that fits a selected actor latent slice to critic ``base_lin_vel``."""

    obs_format: dict[str, dict[str, tuple[int, ...]]]
    obs_group_time_slice_map: dict[str, dict[str, ObsSelector]]
    target_obs_group_name: str = "critic"
    target_obs_term_names: list[str] = field(default_factory=lambda: ["base_lin_vel"])
    loss_name: str = "estimation_loss"
    target_obs_selector: ObsSelector = field(default_factory=lambda: ObsSelector(slice(0, 0)), init=False, repr=False)
    estimation_latent_selector: slice = field(default_factory=lambda: slice(0, 0), init=False, repr=False)

    def validate(self, algo: object) -> None:
        """Validate the actor assumptions and cache the resolved target/latent slices."""
        actor = getattr(algo, "actor", None)

        if not isinstance(actor, EncoderMLPModel):
            raise ValueError(
                f"`EncoderEstimationLossSpec` requires `algo.actor` to be `EncoderMLPModel`, got {type(actor)}."
            )

        self.target_obs_selector = resolve_target_obs_term_selector(
            target_obs_group_name=self.target_obs_group_name,
            target_obs_term_names=self.target_obs_term_names,
            obs_group_time_slice_map=self.obs_group_time_slice_map,
            obs_format=self.obs_format,
        )
        self.estimation_latent_selector = slice(0, self.target_obs_selector.dim)

        actor_latent_dim = actor.get_latent_dim()
        if actor_latent_dim < self.target_obs_selector.dim:
            raise ValueError(
                "`EncoderEstimationLossSpec` can not infer an estimation latent slice because the actor latent is smaller "
                f"than target_dim={self.target_obs_selector.dim}: actor_latent_dim={actor_latent_dim}."
            )

        print("[EncoderEstimationPPO] Resolved estimation loss target_obs_selector:", self.target_obs_selector)

    def compute(
        self,
        algo: object,
        minibatch: RolloutStorage.Batch,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute an MSE loss between the selected actor latent slice and critic ``base_lin_vel``."""
        actor = algo.actor  # type: ignore[attr-defined]
        actor_latent = actor.get_latent(minibatch.observations)
        actor_estimation_latent = actor_latent[..., self.estimation_latent_selector]
        target_obs = self.target_obs_selector.select(minibatch.observations[self.target_obs_group_name])
        estimation_loss = F.mse_loss(actor_estimation_latent, target_obs)
        return {self.loss_name: estimation_loss}, {}


class EncoderEstimationPPO(ComposablePPO):
    """Composable PPO variant with ``EncoderEstimationLossSpec`` installed."""

    def __init__(self, *args, estimation_loss_coef: float = 1.0, **kwargs) -> None:
        """Initialize the variant and expose an explicit coefficient for ``estimation_loss``."""
        self.estimation_loss_coef = estimation_loss_coef
        super().__init__(*args, **kwargs)

    @classmethod
    def build_loss_spec(cls, env: VecEnv, algorithm_cfg: dict) -> EncoderEstimationLossSpec:
        """Build the encoder-estimation loss spec from environment metadata and algorithm config."""
        if not hasattr(env, "obs_format") or not hasattr(env, "obs_group_layout_mode_map"):
            raise ValueError(
                f"`{cls.__name__}` requires `env` to expose `obs_format` and `obs_group_layout_mode_map`."
            )

        return EncoderEstimationLossSpec(
            obs_format=env.obs_format,
            obs_group_time_slice_map=env.obs_group_time_slice_map,
            target_obs_group_name=algorithm_cfg.pop("target_obs_group_name", "critic"),
            target_obs_term_names=algorithm_cfg.pop("target_obs_term_names", ["base_lin_vel"]),
        )
