# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Derived from FlashSAC, Copyright (c) 2026 Holiday Robotics, under the MIT license.

from __future__ import annotations

import copy
import math
import torch
from tensordict import TensorDict
from torch import Tensor, nn

from z_rl.env import VecEnv
from z_rl.models.composition.adapters import ObsLatentAdapter
from z_rl.modules.normalization import EmpiricalNormalization
from z_rl.utils import resolve_obs_groups

from .sac_base import SACBase, resolve_action_bounds
from .flash_sac_networks import FlashSACActor, FlashSACCritic, normalize_parameters


class FlashSAC(SACBase):
    """Single-device FlashSAC with a one-step categorical Bellman target."""

    def __init__(
        self,
        obs: TensorDict,
        env: VecEnv,
        obs_groups: dict[str, list[str]],
        device: str,
        *,
        clip_actions: float | None = None,
        replay_buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,
        batch_size: int = 2048,
        updates_per_iteration: int = 1,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 256,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
    ) -> None:
        """Build the policy, critics and vector replay on one device."""
        capacity = replay_buffer_size // env.num_envs
        super().__init__(capacity, env.num_envs, device, learning_starts, batch_size, updates_per_iteration)
        self.obs_keys = list(dict.fromkeys(obs_groups["actor"] + obs_groups["critic"]))
        if any(obs[key].ndim != 2 for key in self.obs_keys):
            raise ValueError("FlashSAC requires flat observation groups.")
        actor_dim, critic_dim = (sum(obs[key].shape[-1] for key in obs_groups[role]) for role in ("actor", "critic"))
        low, high = resolve_action_bounds(env, clip_actions)
        self.actor = FlashSACActor(obs_groups["actor"], actor_dim, actor_hidden_dim, low, high).to(device)
        self.critic_obs = ObsLatentAdapter(obs_groups["critic"], nn.Identity())
        self.return_limit = 5.0
        support = torch.linspace(-self.return_limit, self.return_limit, 101)
        self.critic = FlashSACCritic(critic_dim, critic_hidden_dim, env.num_actions, support).to(device)
        for network in (self.actor, self.critic):
            normalize_parameters(network)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.01), device=device))
        self.target_entropy = 0.5 * env.num_actions * math.log(2 * math.pi * math.e * 0.15**2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=learning_rate)
        self.gamma = gamma

        self.return_rms = EmpiricalNormalization(1).to(device)
        self.discounted_return = torch.zeros(env.num_envs, device=device)
        self.max_abs_return = torch.zeros((), device=device)
        weights = torch.arange(1, 17, dtype=torch.float32, device=device).pow(-2)
        self.noise_cdf = (weights / weights.sum()).cumsum(0)
        self.noise = torch.zeros(env.num_envs, env.num_actions, device=device)
        self.noise_steps_left = torch.zeros((), dtype=torch.long, device=device)

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> FlashSAC:
        """Resolve observation groups and construct the algorithm for the runner."""
        if cfg.get("multi_gpu"):
            raise ValueError("FlashSAC supports single-device training.")
        groups = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        options = {key: value for key, value in cfg["algorithm"].items() if key != "class_name"}
        return cls(obs, env, groups, device, clip_actions=cfg.get("clip_actions"), **options)

    @torch.no_grad()
    def act(self, obs: TensorDict) -> Tensor:
        """Cache the transition input and sample actions in environment units."""
        self._obs = obs.select(*self.obs_keys).clone()
        if self.replay.size * self.replay.num_envs < self.learning_starts:
            self._actions = torch.empty_like(self.noise).uniform_(-1, 1)
        else:
            mean, log_std = self.actor.distribution(self.actor.observations(self._obs), training=False)
            std = log_std.exp()
            self.actor.output_std.copy_(std.mean(0))
            refresh = self.noise_steps_left == 0
            period = torch.searchsorted(self.noise_cdf, torch.rand((), device=mean.device))
            self.noise.copy_(torch.where(refresh, torch.randn_like(self.noise), self.noise))
            self.noise_steps_left.copy_(torch.where(refresh, period, self.noise_steps_left - 1))
            self._actions = (mean + std * self.noise).tanh()
        return self.actor.to_env_actions(self._actions)

    @torch.no_grad()
    def process_env_step(self, obs: TensorDict, rewards: Tensor, dones: Tensor, extras: dict) -> None:
        """Store the transition and update the running scale of rewards."""
        transition = self._transition(obs, rewards, dones, extras)
        self.replay.add(transition)
        self.discounted_return.mul_(self.gamma * ~transition["segment_end"]).add_(transition["rewards"])
        torch.maximum(self.max_abs_return, self.discounted_return.abs().max(), out=self.max_abs_return)
        self.return_rms.update(self.discounted_return[:, None])

    def update(self) -> dict[str, float]:
        """Optimize the policy and critics after replay warm-up."""
        if not self._ready_to_update():
            return {}
        metrics = {}
        for _ in range(self.updates_per_iteration):
            batch = self.replay.sample_starts(self.batch_size)
            obs, next_obs = batch["obs"], batch["next_obs"]
            actions = batch["actions"]
            actor_obs, actor_next_obs = self.actor.observations(obs), self.actor.observations(next_obs)
            critic_obs, critic_next_obs = self.critic_obs(obs), self.critic_obs(next_obs)
            return_std = self.return_rms.std.square().add(1e-8).sqrt()
            reward_scale = torch.maximum(return_std, self.max_abs_return / self.return_limit)
            reward = batch["rewards"] / reward_scale
            discount = self.gamma * batch["bootstrap"]

            # FlashSAC updates the actor and temperature before evaluating the critic target.
            if self.update_step % 2 == 0:
                policy_actions, log_prob = self.actor.sample(torch.cat((actor_obs, actor_next_obs)), training=True)
                policy_actions, log_prob = policy_actions[: self.batch_size], log_prob[: self.batch_size]
                self.critic.requires_grad_(False)
                q_values, _ = self.critic(critic_obs, policy_actions, training=False)
                actor_loss = (log_prob * self.log_temperature.detach().exp() - q_values.amin(0)).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()
                normalize_parameters(self.actor)
                self.critic.requires_grad_(True)

                temperature_loss = self.log_temperature.exp() * (-log_prob.detach().mean() - self.target_entropy)
                self.temperature_optimizer.zero_grad(set_to_none=True)
                temperature_loss.backward()
                self.temperature_optimizer.step()
                metrics.setdefault("actor", []).append(actor_loss.detach())

            # BatchNorm sees paired current and next observations on both critics.
            all_obs = torch.cat((critic_obs, critic_next_obs))
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.sample(actor_next_obs, training=False)
                all_actions = torch.cat((actions, next_actions))
                next_values, next_log_probs = self.target_critic(all_obs, all_actions, training=True)
                min_critic = next_values[:, self.batch_size :].argmin(0)
                rows = torch.arange(self.batch_size, device=actions.device)
                next_probs = next_log_probs[min_critic, rows + self.batch_size].exp()
                entropy_term = self.log_temperature.exp() * next_log_prob
                targets = _project_distribution(next_probs, reward, discount, entropy_term, self.critic.support)

            _, log_probs = self.critic(all_obs, all_actions, training=True)
            critic_loss = -(targets * log_probs[:, : self.batch_size]).sum(-1).mean()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()
            normalize_parameters(self.critic)
            with torch.no_grad():
                for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                    target.lerp_(source, 0.01)
            self.update_step += 1
            metrics.setdefault("critic", []).append(critic_loss.detach())

        return {
            **{key: torch.stack(values).mean().item() for key, values in metrics.items()},
            "temperature": self.log_temperature.detach().exp().item(),
        }

    def train_mode(self) -> None:
        """Enable training and reward statistics updates."""
        self.actor.train()
        self.return_rms.train()

    def eval_mode(self) -> None:
        """Switch the policy and reward normalizer to evaluation mode."""
        self.actor.eval()
        self.return_rms.eval()

    def save(self) -> dict:
        """Pack the model and training state for the runner checkpoint."""
        return {
            **super().save(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "temperature_optimizer_state_dict": self.temperature_optimizer.state_dict(),
            "log_temperature": self.log_temperature.detach().clone(),
            "return_rms_state_dict": self.return_rms.state_dict(),
            "max_abs_return": self.max_abs_return.clone(),
        }

    def _load_extra_state(self, checkpoint: dict, load_cfg: dict, strict: bool) -> None:
        if load_cfg.get("critic"):
            self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.temperature_optimizer.load_state_dict(checkpoint["temperature_optimizer_state_dict"])
            self.log_temperature.copy_(checkpoint["log_temperature"])
            self.return_rms.load_state_dict(checkpoint["return_rms_state_dict"], strict=strict)
            self.max_abs_return.copy_(checkpoint["max_abs_return"])
        if load_cfg.get("iteration"):
            self.discounted_return.zero_()
            self.noise_steps_left.zero_()


def _project_distribution(probs: Tensor, reward: Tensor, discount: Tensor, entropy: Tensor, support: Tensor) -> Tensor:
    values = reward[:, None] + discount[:, None] * (support - entropy[:, None])
    last_bin = support.numel() - 1
    indices = ((values - support[0]) * (last_bin / (support[-1] - support[0]))).clamp(0, last_bin)
    lower = indices.floor().long()
    upper = (lower + 1).clamp_max(last_bin)
    frac = indices - lower
    targets = torch.zeros_like(probs)
    targets.scatter_add_(1, lower, probs * (1 - frac))
    return targets.scatter_add_(1, upper, probs * frac)
