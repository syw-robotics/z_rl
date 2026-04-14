"""Algorithm template sources for plugin scaffold generation."""

ALGORITHMS_INIT_TEMPLATE = "from .my_ppo import MyPPO\n\n__all__ = [\"MyPPO\"]\n"

ALGORITHMS_MY_PPO_TEMPLATE = """from __future__ import annotations

import torch

from z_rl.env import VecEnv
from z_rl.algorithms.composition import ComposablePPO, PPOLossSpec
from z_rl.storage import RolloutStorage


class MyAuxLossSpec(PPOLossSpec):
    \"\"\"Example PPO loss spec with one extra loss term.\"\"\"

    def validate(self, algo) -> None:
        return None

    def compute(
        self,
        algo,
        minibatch: RolloutStorage.Batch,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        dummy_loss = torch.zeros((), device=minibatch.actions.device)
        return {\"my_aux_loss\": dummy_loss}, {}


class MyPPO(ComposablePPO):
    \"\"\"Example PPO variant that only needs to define its loss spec.\"\"\"

    @classmethod
    def build_loss_spec(cls, env: VecEnv, algorithm_cfg: dict) -> PPOLossSpec:
        return MyAuxLossSpec()

    def __init__(self, *args, my_aux_loss_coef: float = 0.0, **kwargs) -> None:
        self.my_aux_loss_coef = my_aux_loss_coef
        super().__init__(*args, **kwargs)
"""
