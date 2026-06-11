# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable

from z_rl.env import VecEnv
from z_rl.models import MLPModel
from z_rl.storage import RolloutStorage
from z_rl.utils import resolve_callable


class Symmetry:
    """Symmetry augmentation and mirror-loss helper for PPO."""

    def __init__(
        self,
        env: VecEnv,
        data_augmentation_func: str | Callable,
        use_data_augmentation: bool = False,
        use_mirror_loss: bool = False,
        mirror_loss_coeff: float = 0.0,
    ) -> None:
        """Resolve and store symmetry configuration."""
        self.env = env
        self.use_data_augmentation = use_data_augmentation
        self.use_mirror_loss = use_mirror_loss
        self.mirror_loss_coeff = mirror_loss_coeff
        self.data_augmentation_func = resolve_callable(data_augmentation_func)

        if not callable(self.data_augmentation_func):
            raise ValueError(f"Symmetry data augmentation function is not callable: {data_augmentation_func}")
        if not (use_data_augmentation or use_mirror_loss):
            print("Symmetry not used for learning. We will use it for logging instead.")

    def augment_batch(self, batch: RolloutStorage.Batch, original_batch_size: int) -> None:
        """Append mirrored observations/actions and repeat rollout tensors to match."""
        if not self.use_data_augmentation:
            return

        batch.observations, batch.actions = self.data_augmentation_func(
            env=self.env,
            obs=batch.observations,
            actions=batch.actions,
        )
        num_aug = int(batch.observations.batch_size[0] / original_batch_size)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)

    def compute_loss(self, actor: MLPModel, batch: RolloutStorage.Batch, original_batch_size: int) -> torch.Tensor:
        """Compute mirror loss on policy means; detach it when it is only logged."""
        if not self.use_data_augmentation:
            batch.observations, _ = self.data_augmentation_func(env=self.env, obs=batch.observations, actions=None)

        mean_actions = actor(batch.observations.detach().clone())
        _, mean_actions_symm = self.data_augmentation_func(
            env=self.env, obs=None, actions=mean_actions[:original_batch_size]
        )

        symmetry_loss = nn.functional.mse_loss(
            mean_actions[original_batch_size:],
            mean_actions_symm.detach()[original_batch_size:],
        )
        return symmetry_loss if self.use_mirror_loss else symmetry_loss.detach()


def resolve_symmetry_config(alg_cfg: dict, env: VecEnv) -> dict:
    """Resolve the symmetry configuration.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        env: Environment object.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    # If using symmetry then pass the environment object
    # Note: This is used by the symmetry function for handling different observation terms
    if "symmetry_cfg" in alg_cfg and alg_cfg["symmetry_cfg"] is not None:
        alg_cfg["symmetry_cfg"]["env"] = env
    else:
        alg_cfg["symmetry_cfg"] = None
    return alg_cfg
