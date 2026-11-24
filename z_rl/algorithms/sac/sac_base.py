# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

"""Replay collection and runner state shared by the SAC algorithms."""

from __future__ import annotations

import math
import torch
from abc import ABC, abstractmethod
from tensordict import TensorDict
from torch import Tensor, nn

from z_rl.env import VecEnv
from z_rl.storage.sequence_replay import SequenceReplayBuffer
from z_rl.utils.checkpoint import model_checkpoint_entries


class SACBase(ABC):
    """Common replay bookkeeping and checkpoint format for SAC and FlashSAC."""

    actor: nn.Module
    critic: nn.Module
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    obs_keys: list[str] | tuple[str, ...]
    _obs: TensorDict
    _actions: Tensor

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        device: str,
        learning_starts: int,
        batch_size: int,
        updates_per_iteration: int,
        n_steps: int = 1,
    ) -> None:
        """Initialize replay and validate the number of transitions needed for an update."""
        if min(batch_size, updates_per_iteration) < 1:
            raise ValueError("SAC batch size and updates per iteration must be positive.")
        if not 1 <= n_steps <= capacity or not 0 <= learning_starts <= capacity * num_envs:
            raise ValueError("The SAC replay capacity must cover n_steps and learning_starts.")
        self.device = device
        self.replay = SequenceReplayBuffer(capacity, num_envs, device)
        self.learning_starts = learning_starts
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.updates_per_iteration = updates_per_iteration
        self.update_step = 0

    def _ready_to_update(self) -> bool:
        return self.replay.size >= self.n_steps and self.replay.size * self.replay.num_envs >= self.learning_starts

    def _transition(self, obs: TensorDict, rewards: Tensor, dones: Tensor, extras: dict) -> dict:
        """Use the terminal observation for bootstrapping a time-limited episode."""
        next_obs = obs.select(*self.obs_keys)
        done = dones.reshape(-1).bool()
        bootstrap = ~done
        if extras.get("time_outs_obs") is not None and "time_outs" in extras:
            timeout = done & extras["time_outs"].to(done).reshape(-1)
            if "terminated" in extras:
                timeout &= ~extras["terminated"].to(done).reshape(-1)
            terminal = TensorDict(extras["time_outs_obs"], batch_size=obs.batch_size).select(*self.obs_keys)
            next_obs = terminal.to(self.device).where(timeout, next_obs)
            bootstrap |= timeout
        return dict(
            obs=self._obs,
            actions=self._actions,
            rewards=rewards.reshape(-1),
            next_obs=next_obs,
            segment_end=done,
            bootstrap=bootstrap,
        )

    def compute_returns(self, obs: TensorDict) -> None:
        """SAC constructs Bellman targets from replay during update()."""

    @property
    def learning_rate(self) -> float:
        """Return the actor learning rate for the runner logger."""
        return self.actor_optimizer.param_groups[0]["lr"]

    def get_policy(self) -> nn.Module:
        """Return the actor for inference."""
        return self.actor

    def save(self) -> dict:
        """Pack the model, optimizer and iteration state shared by both algorithms."""
        return {
            **model_checkpoint_entries("actor", self.actor),
            **model_checkpoint_entries("critic", self.critic),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    @torch.no_grad()
    def load(self, checkpoint: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Restore selected state and start with empty replay when resuming an iteration."""
        if load_cfg is None:
            load_cfg = dict(actor=True, critic=True, optimizer=True, iteration=True)
        if load_cfg.get("actor"):
            self.actor.load_state_dict(checkpoint["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(checkpoint["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self._load_extra_state(checkpoint, load_cfg, strict)
        if load_cfg.get("iteration"):
            self.update_step = checkpoint["update_step"]
            self.replay.size = self.replay.position = 0
        return load_cfg.get("iteration", False)

    @abstractmethod
    def _load_extra_state(self, checkpoint: dict, load_cfg: dict, strict: bool) -> None:
        """Restore the algorithm-specific portion of the selected checkpoint state."""


def resolve_action_bounds(
    env: VecEnv, clip_actions: float | None, *, action_low=None, action_high=None
) -> tuple[Tensor, Tensor]:
    """Resolve per-action limits in VecEnv units, including clipping and policy overrides."""
    space = getattr(getattr(env, "unwrapped", env), "single_action_space", None)
    if space is None:
        space = getattr(env, "action_space", None)
    clip = getattr(env, "clip_actions", None)
    clip = clip if clip is not None else clip_actions
    bounds = []
    for name, sign, override in (("low", -1, action_low), ("high", 1, action_high)):
        bound = torch.as_tensor(getattr(space, name, sign * math.inf), dtype=torch.float32, device="cpu")
        if bound.ndim == 2:
            if not (bound == bound[0]).all():
                raise ValueError("SAC requires common action bounds across vectorized environments.")
            bound = bound[0]
        if clip is not None:
            bound = bound.clamp(min=-clip) if sign < 0 else bound.clamp(max=clip)
        if override is None:
            bound = torch.where(bound.isfinite(), bound, float(sign))
        else:
            override = torch.as_tensor(override, device="cpu")
            if (sign * override > sign * bound).any():
                raise ValueError(f"action_{name} exceeds the environment action bounds.")
            bound = override.to(torch.float32)
        bounds.append(bound.expand(env.num_actions))
    low, high = bounds
    if (low >= high).any():
        raise ValueError("SAC requires action bounds with positive width.")
    return low, high
