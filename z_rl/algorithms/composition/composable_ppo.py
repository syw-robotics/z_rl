# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from tensordict import TensorDict

from z_rl.env import VecEnv
from z_rl.storage import RolloutStorage

from ..ppo import PPO
from .specs import PPOLossSpec


class ComposablePPO(PPO):
    """PPO variant that explicitly applies one optional loss spec after base loss computation."""

    def __init__(self, *args, loss_spec: PPOLossSpec | None = None, **kwargs) -> None:
        """Initialize PPO and store the optional additional loss spec."""
        super().__init__(*args, **kwargs)
        self.loss_spec = loss_spec
        if self.loss_spec is not None:
            self.loss_spec.validate(self)
        else:
            raise ValueError("`ComposablePPO` requires a `loss_spec` to be provided for additional loss computation.")

    def compute_loss(self, minibatch: RolloutStorage.Batch) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute base PPO loss and merge any additional losses from the configured spec."""
        opt_losses, non_opt_losses = super().compute_loss(minibatch)
        if self.loss_spec is not None:
            extra_opt_losses, extra_non_opt_losses = self.loss_spec.compute(self, minibatch)
            opt_losses.update(extra_opt_losses)
            non_opt_losses.update(extra_non_opt_losses)
        return opt_losses, non_opt_losses

    def act(self, obs: TensorDict) -> torch.Tensor:
        """ Subclasses can override this method when a PPO variant needs full control over rollout-time action generation
        and transition bookkeeping.
        """
        return super().act(obs)

    def forward_actor_for_update(self, minibatch: RolloutStorage.Batch) -> None:
        """Subclasses can override update-time actor forwards."""
        super().forward_actor_for_update(minibatch)

    @classmethod
    def build_loss_spec(cls, env: VecEnv, algorithm_cfg: dict) -> PPOLossSpec:
        """Build the loss spec for this PPO variant from the environment and algorithm config."""
        raise NotImplementedError(f"`{cls.__name__}` must override `build_loss_spec()`.")

    @classmethod
    def _build_algorithm_extra_kwargs(cls, env: VecEnv, algorithm_cfg: dict) -> dict:
        """Build composable PPO-specific keyword arguments for shared PPO construction."""
        return {"loss_spec": cls.build_loss_spec(env, algorithm_cfg)}
