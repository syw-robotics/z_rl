# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rl_cfg import ZRlBaseRunnerCfg, ZRlMLPModelCfg

############################
# Algorithm configurations #
############################


@configclass
class ZRlDistillationAlgorithmCfg:
    """Configuration for the distillation algorithm."""

    class_name: str = "Distillation"
    """The algorithm class name. Defaults to Distillation."""

    num_learning_epochs: int = MISSING
    """The number of updates performed with each sample."""

    learning_rate: float = MISSING
    """The learning rate for the student policy."""

    gradient_length: int = MISSING
    """The number of environment steps the gradient flows back."""

    max_grad_norm: None | float = None
    """The maximum norm the gradient is clipped to. Defaults to None."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop", "muon_adamw"] = "adam"
    """Optimizer for the student. ``"muon_adamw"`` splits eligible 2-D weights from the remaining parameters."""

    use_muon: bool = False
    """Legacy flag. When True, selects ``"muon_adamw"`` regardless of ``optimizer``."""

    muon_adamw_weight_decay: float = 0.01
    """Weight decay for the AdamW group in ``"muon_adamw"``."""

    loss_type: Literal["mse", "huber"] = "mse"
    """The loss type to use for the student policy. Defaults to mse."""

    student_stochastic_output: bool = False
    """Whether to use stochastic student outputs during rollout and distillation updates. Defaults to False."""

    use_mixed_precision: bool = False
    """Whether to run the forward pass and loss computation in bfloat16 autocast. Defaults to False.

    Backward, gradient clipping, and the optimizer step always stay in fp32.
    """

    grad_reduce_bucket_mb: float = 25.0
    """Maximum size, in megabytes, of a packed gradient buffer during multi-GPU reduction. Defaults to 25.

    Matches ``torch.nn.parallel.DistributedDataParallel``'s default ``bucket_cap_mb``.
    """


#########################
# Runner configurations #
#########################


@configclass
class ZRlDistillationRunnerCfg(ZRlBaseRunnerCfg):
    """Configuration of the runner for distillation algorithms."""

    class_name: str = "DistillationRunner"
    """The runner class name. Defaults to DistillationRunner."""

    student: ZRlMLPModelCfg = MISSING
    """The student configuration."""

    teacher: ZRlMLPModelCfg = MISSING
    """The teacher configuration."""

    algorithm: ZRlDistillationAlgorithmCfg = MISSING
    """The algorithm configuration."""
