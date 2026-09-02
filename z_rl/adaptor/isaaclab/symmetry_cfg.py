# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


@configclass
class ZRlSymmetryCfg:
    """Optional mirror-loss and diagnostic settings for symmetry-aware training.

    Rollout data augmentation is controlled exclusively by
    :attr:`ZRlPpoAlgorithmCfg.symmetry_augmentation`. Observation and action transforms are declared by the
    environment and compiled by its z_rl adaptor.

    When :meth:`use_mirror_loss` is True, the :meth:`mirror_loss_coeff` is used to weight the
    symmetry-mirror loss. This loss is directly added to the agent's loss function.

    :meth:`log_mirror_loss` independently enables a low-frequency detached diagnostic. When both switches are
    False, this config adds no mirror-loss or metric-forward computation.

    For more information, please check the work from :cite:`mittal2024symmetry`.
    """

    use_mirror_loss: bool = False
    """Whether to use the symmetry-augmentation loss. Defaults to False."""

    log_mirror_loss: bool = False
    """Whether to compute a detached mirror-loss diagnostic when mirror loss is disabled. Defaults to False."""

    mirror_loss_log_interval: int = 100
    """PPO update interval for the detached mirror-loss diagnostic. Defaults to 100.

    The metric is computed on only the first mini-batch of a selected update.
    """

    mirror_loss_coeff: float = 0.0
    """The weight for the symmetry-mirror loss. Defaults to 0.0."""
