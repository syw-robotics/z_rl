"""Algorithm template sources for plugin scaffold generation."""

ALGORITHMS_INIT_TEMPLATE = "from .my_ppo import MyPPO\n\n__all__ = [\"MyPPO\"]\n"

ALGORITHMS_MY_PPO_TEMPLATE = """from __future__ import annotations

import torch

from z_rl.algorithms import PPO
from z_rl.algorithms.mixins.ppo_mixin import BasePPOMixin
from z_rl.storage import RolloutStorage


class MyPPO(BasePPOMixin, PPO):
    \"\"\"Example PPO extension with one extra loss hook.\"\"\"

    def __init__(self, *args, my_aux_loss_coef: float = 0.0, **kwargs) -> None:
        self.my_aux_loss_coef = my_aux_loss_coef
        super().__init__(*args, **kwargs)

    def compute_additional_loss(
        self, minibatch: RolloutStorage.Batch
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        dummy_loss = torch.zeros((), device=minibatch.actions.device)
        return {\"my_aux_loss\": dummy_loss}, {}
"""
