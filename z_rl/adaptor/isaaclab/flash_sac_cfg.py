# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab baseline preset for FlashSAC."""

from dataclasses import field

from isaaclab.utils import configclass

from .rl_cfg import ZRlBaseRunnerCfg


@configclass
class ZRlFlashSacAlgorithmCfg:
    """Settings for the single-device, one-step FlashSAC update."""

    class_name: str = "z_rl.algorithms.flash_sac:FlashSAC"
    replay_buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 2048
    updates_per_iteration: int = 1
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    gamma: float = 0.99
    learning_rate: float = 3e-4


@configclass
class ZRlFlashSacRunnerCfg(ZRlBaseRunnerCfg):
    """Run one FlashSAC training iteration per vectorized environment step."""

    class_name: str = "OnPolicyRunner"
    num_steps_per_env: int = 1
    experiment_name: str = "flash_sac"
    save_interval: int = 1000
    obs_groups: dict[str, list[str]] = field(default_factory=lambda: {"actor": ["policy"]})
    algorithm: ZRlFlashSacAlgorithmCfg = field(default_factory=ZRlFlashSacAlgorithmCfg)
