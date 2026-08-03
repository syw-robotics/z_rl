from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd

from z_rl.modules.mlp import MLP


class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        amp_reward_coef: float = 1.0,
        hidden_dims: list[int] = [256, 128],
        activation: str = "relu",
        task_reward_lerp: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        self.net = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
        )

    def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, next_state], dim=-1))

    @torch.no_grad()
    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()

        d = self.forward(state, next_state)
        amp_reward = self.amp_reward_coef * torch.clamp(
            1.0 - 0.25 * torch.square(d - 1.0),
            min=0.0,
        )

        if self.task_reward_lerp > 0.0:
            task_reward = task_reward.view_as(amp_reward)
            reward = (1.0 - self.task_reward_lerp) * amp_reward + self.task_reward_lerp * task_reward
        else:
            reward = amp_reward

        if was_training:
            self.train()

        return reward.squeeze(-1), d

    def compute_grad_pen(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        lambda_: float = 10.0,
    ) -> torch.Tensor:
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data = expert_data.detach()
        expert_data.requires_grad_(True)

        d = self.net(expert_data)
        ones = torch.ones_like(d)

        grad = autograd.grad(
            outputs=d,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return lambda_ * (grad.norm(2, dim=1) - 0.0).pow(2).mean()