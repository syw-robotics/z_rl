# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

"""Soft Actor-Critic for continuous actions and vectorized, auto-resetting environments."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict
from torch import nn

from z_rl.env import VecEnv
from z_rl.models import MLPModel, SACCriticModel
from z_rl.models.construction import build_model, init_cnn_weights, init_head_weights
from z_rl.modules import SquashedGaussianDistribution
from z_rl.utils import (
    reduce_gradients_in_buckets,
    resolve_callable,
    resolve_obs_groups,
)
from z_rl.utils.opt import build_optimizer

from .sac_base import SACBase, resolve_action_bounds


class SAC(SACBase):
    """Train a reparameterized actor and twin Q heads from vectorized replay."""

    def __init__(
        self,
        actor: MLPModel,
        critic: SACCriticModel,
        num_envs: int,
        num_actions: int,
        device: str,
        *,
        replay_buffer_size=1_000_000,
        learning_starts=1024,
        batch_size=256,
        updates_per_iteration=1,
        n_steps=1,
        gamma=0.998,
        tau=0.005,
        policy_frequency=2,
        max_grad_norm=1.0,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        alpha_learning_rate=1e-3,
        actor_optimizer="adam",
        critic_optimizer="adam",
        alpha=0.05,
        auto_alpha=True,
        target_entropy_scale=1.0,
        multi_gpu_cfg=None,
    ) -> None:
        if replay_buffer_size < 1 or policy_frequency < 1:
            raise ValueError("SAC replay size and policy frequency must be positive.")
        capacity = max(1, replay_buffer_size // num_envs)
        super().__init__(capacity, num_envs, device, learning_starts, batch_size, updates_per_iteration, n_steps)
        if not 0 <= gamma <= 1 or not 0 < tau <= 1 or alpha <= 0:
            raise ValueError("SAC requires gamma in [0, 1], tau in (0, 1] and positive alpha.")
        self.actor, self.critic = actor, critic
        self.obs_keys = tuple(dict.fromkeys(actor.obs_groups + critic.obs_groups))
        self.discounts = gamma ** torch.arange(n_steps + 1, device=device, dtype=torch.float32)
        self.policy_frequency = policy_frequency
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg else 1
        self.actor_optimizer = build_optimizer([actor], learning_rate=actor_learning_rate, optimizer=actor_optimizer)
        self.critic_optimizer = build_optimizer(
            [critic.head, critic.latent_adapter], learning_rate=critic_learning_rate, optimizer=critic_optimizer
        )
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha), device=device), requires_grad=auto_alpha)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate) if auto_alpha else None
        self.target_entropy = -num_actions * target_entropy_scale

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SAC:
        groups = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        actor_cfg = cfg["actor"]
        actor_class = resolve_callable(actor_cfg["class_name"])
        dist_cfg = dict(actor_cfg.get("distribution_cfg") or {"class_name": "SquashedGaussianDistribution"})
        if actor_class.is_recurrent or not issubclass(
            resolve_callable(dist_cfg["class_name"]), SquashedGaussianDistribution
        ):
            raise ValueError("SAC requires a feedforward actor with SquashedGaussianDistribution.")
        dist_cfg["action_low"], dist_cfg["action_high"] = resolve_action_bounds(
            env, cfg.get("clip_actions"), action_low=dist_cfg.get("action_low"), action_high=dist_cfg.get("action_high")
        )
        actor = build_model(actor_cfg, obs, groups, "actor", env.num_actions, env, device, distribution_cfg=dist_cfg)
        if actor_cfg.get("init_weights") is not None:
            init_head_weights(actor, actor_cfg, "Actor")
            actor.distribution.init_head_weights(actor.head)
        init_cnn_weights(actor, actor_cfg, "Actor")
        critic_cfg = cfg["critic"]
        # Initialize the online Q heads before the critic copies them to the targets.
        critic_kwargs = {"init_weights": critic_cfg["init_weights"]} if "init_weights" in critic_cfg else {}
        critic = build_model(
            critic_cfg, obs, groups, "critic", 1, env, device, num_actions=env.num_actions, **critic_kwargs
        )
        init_cnn_weights(critic, critic_cfg, "Critic")
        algo_cfg = dict(cfg["algorithm"])
        algo_cfg.pop("class_name")
        return cls(actor, critic, env.num_envs, env.num_actions, device, multi_gpu_cfg=cfg.get("multi_gpu"), **algo_cfg)

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        # A VecEnv may reuse its observation and action tensors during step().
        self._obs = obs.select(*self.obs_keys).clone()
        actions = self.actor(self._obs, stochastic_output=True)
        self._actions = actions.clone()
        return actions

    @torch.no_grad()
    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        transition = self._transition(obs, rewards, dones, extras)
        self._store_transition(transition, extras)
        self.actor.update_normalization(self._obs)
        self.critic.update_normalization(self._obs)

    def _store_transition(self, transition, extras):
        self.replay.add(transition)

    def _sample_replay(self, replay, batch_size):
        window = TensorDict(replay.sample_windows(batch_size, self.n_steps), [batch_size, self.n_steps]).to(self.device)
        mask = window["mask"]
        length = mask.sum(1).long()
        rows = torch.arange(batch_size, device=self.device)
        last = (rows, length - 1)
        weights = mask * self.discounts[:-1]
        discount = self.discounts[length] * window["bootstrap"][last]
        if "env_discount" in window:
            factors = torch.cat((torch.ones_like(mask[:, :1]), window["env_discount"]), dim=1).cumprod(dim=1)
            weights = weights * factors[:, :-1]
            discount = discount * factors[rows, length]
        batch = window[:, 0].exclude("mask", "segment_end", "bootstrap", "env_discount")
        batch["rewards"] = (weights * window["rewards"]).sum(1)
        batch["next_obs"] = window["next_obs"][last]
        batch["discount"] = discount
        return batch

    def _sample_batch(self):
        return self._sample_replay(self.replay, self.batch_size)

    def _sample_policy(self, obs):
        dist = self.actor.distribution
        dist.update(self.actor.head(self.actor.get_latent(obs)))
        return dist.rsample_with_log_prob()

    def _optimize(self, loss, optimizer, *, clip_grad=True):
        optimizer.zero_grad(set_to_none=True)
        params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
        loss.backward(inputs=params)
        if self.world_size > 1:
            reduce_gradients_in_buckets(params, self.world_size, bucket_mb=25)
        if clip_grad:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()

    def _policy_schedule(self):
        return self.update_step % self.policy_frequency == 0, self.alpha_optimizer is not None

    def _mean_loss(self, per_row, batch):
        return per_row.mean()

    def _actor_loss(self, batch, action, logp, metrics):
        q_policy = self.critic(batch["obs"], action).amin(-1)
        return (self.log_alpha.detach().exp() * logp - q_policy).mean()

    def update(self) -> dict[str, float]:
        if not self._ready_to_update():
            return {}
        metrics = {}
        for _ in range(self.updates_per_iteration):
            batch = self._sample_batch()
            with torch.no_grad():
                next_action, next_logp = self._sample_policy(batch["next_obs"])
                next_q = self.critic(batch["next_obs"], next_action, target=True).amin(-1)
                target = batch["rewards"] + batch["discount"] * (next_q - self.log_alpha.exp() * next_logp)

            q = self.critic(batch["obs"], batch["actions"])
            critic_loss = self._mean_loss((q - target.unsqueeze(-1)).square().mean(-1), batch)
            self._optimize(critic_loss, self.critic_optimizer)
            metrics.setdefault("critic", []).append(critic_loss.detach())

            update_actor, update_alpha = self._policy_schedule()
            if update_actor or update_alpha:
                with torch.set_grad_enabled(update_actor):
                    action, logp = self._sample_policy(batch["obs"])
                if update_alpha:
                    alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                    self._optimize(alpha_loss, self.alpha_optimizer, clip_grad=False)
                if update_actor:
                    actor_loss = self._actor_loss(batch, action, logp, metrics)
                    self._optimize(actor_loss, self.actor_optimizer)
                    metrics.setdefault("actor", []).append(actor_loss.detach())
            self.critic.update_target(self.tau)
            self.update_step += 1

        return {
            **{key: torch.stack(values).mean().item() for key, values in metrics.items()},
            "alpha": self.log_alpha.detach().exp().item(),
        }

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()

    @torch.no_grad()
    def broadcast_parameters(self) -> None:
        for tensor in (*self.actor.state_dict().values(), *self.critic.state_dict().values(), self.log_alpha):
            torch.distributed.broadcast(tensor, src=0)

    def save(self) -> dict:
        return {
            **super().save(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            "log_alpha": self.log_alpha.detach().clone(),
        }

    def _load_extra_state(self, checkpoint: dict, load_cfg: dict, strict: bool) -> None:
        if load_cfg.get("optimizer"):
            self.log_alpha.copy_(checkpoint["log_alpha"])
            if self.alpha_optimizer and checkpoint["alpha_optimizer_state_dict"] is not None:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
