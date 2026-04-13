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
from z_rl.utils import get_obs_time_selector_dim, resolve_obs_component_selector, select_obs_component

from ..composition import ComposablePPO, PPOLossSpec


@dataclass(slots=True)
class EncoderEstimationLossSpec(PPOLossSpec):
    """PPO loss spec that fits a selected actor latent slice to critic ``base_lin_vel``."""

    obs_format: dict[str, dict[str, tuple[int, ...]]]
    obs_group_layout_mode_map: dict[str, str]
    target_obs_group_name: str = "critic"
    term_name: str = "base_lin_vel"
    latent_selector: slice | torch.Tensor = field(default_factory=lambda: slice(0, 3))
    loss_name: str = "estimation_loss"
    target_selector: slice | torch.Tensor | None = field(default=None, init=False, repr=False)

    def validate(self, algo: object) -> None:
        """Validate that the actor is an ``EncoderMLPModel`` and cache the target selector."""
        actor = getattr(algo, "actor", None)
        critic = getattr(algo, "critic", None)

        if not isinstance(actor, EncoderMLPModel):
            raise ValueError(
                f"`EncoderEstimationLossSpec` requires `algo.actor` to be `EncoderMLPModel`, got {type(actor)}."
            )
        if critic is None or not hasattr(critic, "obs_dim"):
            raise ValueError("`EncoderEstimationLossSpec` requires the critic model to expose observation metadata.")

        self.target_selector = resolve_obs_component_selector(
            obs_group_name=self.target_obs_group_name,
            term_name=self.term_name,
            obs_format=self.obs_format,
            obs_group_layout_mode_map=self.obs_group_layout_mode_map,
            frame="last",
        )
        latent_dim = get_obs_time_selector_dim(self.latent_selector, actor.get_latent_dim())
        target_dim = get_obs_time_selector_dim(self.target_selector, critic.obs_dim)
        if target_dim != latent_dim:
            raise ValueError(
                "`EncoderEstimationLossSpec` requires the selected latent and target component dims to match, "
                f"got latent_dim={latent_dim}, target_dim={target_dim} for '{self.target_obs_group_name}/{self.term_name}'."
            )

    def compute(
        self,
        algo: object,
        minibatch: RolloutStorage.Batch,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute an MSE loss between the selected actor latent slice and critic ``base_lin_vel``."""
        actor = algo.actor  # type: ignore[attr-defined]
        actor_hidden_state = minibatch.hidden_states[0] if minibatch.hidden_states is not None else None
        actor_latent = actor.get_latent(minibatch.observations, masks=minibatch.masks, hidden_state=actor_hidden_state)
        target = select_obs_component(minibatch.observations[self.target_obs_group_name], self.target_selector)  # type: ignore[index]
        estimation_loss = F.mse_loss(actor_latent[..., self.latent_selector], target)
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
            obs_group_layout_mode_map=env.obs_group_layout_mode_map,
            target_obs_group_name=algorithm_cfg.pop("obs_group_name", "critic"),
            term_name=algorithm_cfg.pop("term_name", "base_lin_vel"),
            latent_selector=algorithm_cfg.pop("latent_selector", slice(0, 3)),
            loss_name=algorithm_cfg.pop("loss_name", "estimation_loss"),
        )
