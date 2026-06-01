# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from z_rl.algorithms import Distillation
from z_rl.runners import OnPolicyRunner


class DistillationRunner(OnPolicyRunner):
    """Distillation runner for student-teacher algorithms."""

    alg: Distillation
    """The distillation algorithm."""

    def print_log_info(self, it: int, start_it: int, total_it: int, collect_time: float, learn_time: float, loss_dict: dict) -> None:
        self.logger.log(
            it=it,
            start_it=start_it,
            total_it=total_it,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=self.alg.learning_rate,
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop after validating that the teacher model is loaded."""
        # Check if teacher is loaded
        if not self.alg.teacher_loaded:
            raise ValueError("[DistillationRunner] Teacher model parameters not loaded. Please load a teacher model to distill.")

        super().learn(num_learning_iterations, init_at_random_ep_len)
