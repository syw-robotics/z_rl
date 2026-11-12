# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass

import torch

from z_rl.modules import MoE
from z_rl.storage import RolloutStorage

from ..composition import ComposablePPO, PPOLossSpec


def _moe_heads(algo: object) -> list[MoE]:
    """Return MoE heads on the raw actor and critic, if present."""
    heads: list[MoE] = []
    for name in ("_raw_actor", "_raw_critic"):
        model = getattr(algo, name, None)
        head = getattr(model, "head", None)
        if isinstance(head, MoE):
            heads.append(head)
    return heads


@dataclass
class MoERoutingLossSpec(PPOLossSpec):
    """PPO loss spec that regularizes MoE gate collapse."""

    gate_entropy_loss_coef: float = 0.0
    expert_balance_loss_coef: float = 1.0e-4

    def validate(self, algo: object) -> None:
        """Require at least one MoE head on the actor or critic, then cache them."""
        self._heads = _moe_heads(algo)
        if not self._heads:
            raise ValueError(
                "`MoERoutingLossSpec` requires the actor or critic head to be `MoE`. "
                "Set `class_name` to `MoEModel` on at least one side."
            )

    def compute(
        self,
        algo: object,
        minibatch: RolloutStorage.Batch,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Average gate-entropy and expert-balance terms across cached MoE heads."""
        # Routing stats come from gate weights cached on the MoE heads during the PPO forward,
        # so the unused ``algo`` / ``minibatch`` arguments are discarded to mark them unused.
        del algo, minibatch

        # Compute the mean gate entropy and expert balance loss across all MoE heads.
        gate_entropy = torch.stack([head.gate_entropy() for head in self._heads]).mean()
        expert_balance_loss = torch.stack([head.expert_balance_loss() for head in self._heads]).mean()

        # Return the loss terms and the routing stats.
        return (
            {
                "gate_entropy_loss": -gate_entropy,
                "expert_balance_loss": expert_balance_loss,
            },
            {"moe_gate_entropy": gate_entropy},
        )


class MoEPPO(ComposablePPO):
    """Named PPO preset that installs ``MoERoutingLossSpec`` by default."""

    loss_spec_class = MoERoutingLossSpec
