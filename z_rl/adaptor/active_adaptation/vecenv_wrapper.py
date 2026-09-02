# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Z-RL vector-environment wrapper for Active Adaptation environments."""

from __future__ import annotations

import torch
import warnings
from collections.abc import Sequence
from tensordict import TensorDict, TensorDictBase
from typing import Any, Literal

from z_rl.env import VecEnv


class ActiveAdaptationVecEnvWrapper(VecEnv):
    """Adapt an AA TorchRL environment to the Z-RL ``VecEnv`` protocol.

    The wrapper owns the current TorchRL carry. Z-RL supplies only a flat action
    tensor, while the wrapper returns reset-aware observations and extracts the
    pre-reset reward, termination, and episode statistics from the transition.
    """

    def __init__(
        self,
        env: Any,
        *,
        observation_keys: Sequence[str],
        reward_keys: Sequence[str] | None,
        action_key: str = "action",
        clip_actions: float | None = None,
        discount_mode: Literal["ignore", "warn", "error"] = "ignore",
        include_episode_stats: bool = False,
    ) -> None:
        """Initialize the wrapper and reset the AA environment once."""
        if not observation_keys:
            raise ValueError(
                "observation_keys must contain at least one AA observation group"
            )
        if reward_keys is not None and not reward_keys:
            raise ValueError(
                "reward_keys must be None or contain at least one AA reward group"
            )
        if discount_mode not in ("ignore", "warn", "error"):
            raise ValueError("discount_mode must be 'ignore', 'warn', or 'error'")
        if clip_actions is not None and clip_actions <= 0:
            raise ValueError("clip_actions must be positive")

        self.env = env
        self.unwrapped = getattr(env, "base_env", env)
        self.observation_keys = tuple(observation_keys)
        self.reward_keys = tuple(reward_keys) if reward_keys is not None else None
        reward_groups = getattr(self.unwrapped, "reward_groups", {})
        self._reward_log_groups = frozenset(reward_groups or self.reward_keys or ())
        self.action_key = action_key
        self.clip_actions = clip_actions
        self.discount_mode = discount_mode
        self.include_episode_stats = include_episode_stats
        self._discount_warning_emitted = False
        self._symmetry_compiler = None

        self.num_envs = int(self.unwrapped.num_envs)
        self.device = self.unwrapped.device
        self.cfg = self.unwrapped.cfg

        input_managers = getattr(self.unwrapped, "input_managers", {})
        if action_key not in input_managers:
            raise KeyError(
                f"AA action input {action_key!r} was not found. Available inputs: {list(input_managers)}"
            )
        self.num_actions = int(input_managers[action_key].action_dim)
        self.max_episode_length = self._resolve_max_episode_length(
            self.unwrapped.max_episode_length
        )

        self._carry = self.env.reset()
        self._validate_observations(self._carry)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Expose AA's episode-length buffer to the Z-RL runner."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        """Seed the wrapped AA environment."""
        return self.env.set_seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:
        """Reset every AA environment and return Z-RL observations."""
        self._carry = self.env.reset()
        self._validate_observations(self._carry)
        return self.get_observations(), {}

    def get_observations(self) -> TensorDict:
        """Return the current AA observation groups selected for Z-RL."""
        return TensorDict(
            {key: self._carry.get(key) for key in self.observation_keys},
            batch_size=[self.num_envs],
            device=self._carry.device,
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Apply a flat Z-RL action and return one reset-aware AA transition."""
        self._validate_actions(actions)
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)

        self._carry.set(self.action_key, actions)
        transition, self._carry = self.env.step_and_maybe_reset(self._carry)
        next_td = transition.get("next")

        rewards = self._aggregate_rewards(next_td.get("reward"))
        done_mask = next_td.get("done").reshape(self.num_envs, -1).any(dim=-1)
        truncated = next_td.get("truncated").reshape(self.num_envs, -1).any(dim=-1)
        discount = next_td.get(
            "discount", torch.ones(self.num_envs, 1, device=self.device)
        ).reshape(self.num_envs, -1)
        self._handle_discount(discount)

        extras: dict = {
            "time_outs": truncated,
            "discount": discount.squeeze(-1) if discount.shape[-1] == 1 else discount,
        }
        if self.include_episode_stats and "stats" in next_td:
            logs, scales = self._prepare_episode_stats(next_td.get("stats"))
            extras["log"] = logs
            extras["log_on_done"] = True
            extras["log_scale"] = scales

        observations = self.get_observations()
        dones = done_mask.to(dtype=torch.long)
        return observations, rewards, dones, extras

    def compile_symmetry(self) -> None:
        """Compile selected AA observation-group and action symmetry transforms."""
        if self._symmetry_compiler is None:
            from .symmetry import ActiveAdaptationSymmetryCompiler

            self._symmetry_compiler = ActiveAdaptationSymmetryCompiler(self)

    def augment_symmetry(
        self,
        obs: TensorDict | None = None,
        actions: torch.Tensor | None = None,
    ) -> tuple[TensorDict | None, torch.Tensor | None]:
        """Append mirrored AA observations and actions along the batch dimension."""
        self.compile_symmetry()
        return self._symmetry_compiler.augment(obs=obs, actions=actions)

    def close(self) -> None:
        """Close the wrapped AA environment."""
        self.env.close()

    def _aggregate_rewards(
        self, rewards: torch.Tensor | TensorDictBase
    ) -> torch.Tensor:
        if isinstance(rewards, TensorDictBase):
            if self.reward_keys is None:
                raise ValueError(
                    "AA returned grouped rewards. Specify reward_keys explicitly when creating the wrapper."
                )
            missing = [key for key in self.reward_keys if key not in rewards]
            if missing:
                raise KeyError(
                    f"AA reward groups {missing} were not found. Available groups: {list(rewards.keys())}"
                )
            reward_tensors = [
                rewards.get(key).reshape(self.num_envs, -1) for key in self.reward_keys
            ]
            reward = torch.cat(reward_tensors, dim=-1).sum(dim=-1)
        else:
            if self.reward_keys is not None:
                raise ValueError(
                    "AA returned a scalar reward tensor, so reward_keys must be None."
                )
            reward = rewards.reshape(self.num_envs, -1).sum(dim=-1)
        return reward

    def _prepare_episode_stats(
        self, stats: TensorDictBase
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Expose device-resident stats; the logger reuses its completed-env IDs."""
        logs = {}
        scales = {}
        for key, value in stats.items(include_nested=True, leaves_only=True):
            path = key if isinstance(key, tuple) else (key,)
            log_key = self._episode_log_key(path)
            logs[log_key] = value.detach()
            if len(path) == 2 and path[0] in self._reward_log_groups:
                # AA sums weighted terms per episode. Z-RL's IsaacLab logs use
                # their contribution per configured episode step.
                scales[log_key] = 1.0 / self.max_episode_length
        return logs, scales

    def _episode_log_key(self, path: tuple[str, ...]) -> str:
        """Map AA stats to the metric namespaces consumed by Z-RL loggers."""
        if len(path) == 2 and path[0] in self._reward_log_groups:
            group = "" if len(self._reward_log_groups) == 1 else f"{path[0]}/"
            return f"Episode_Reward/{group}{path[1]}"
        if len(path) == 2 and path[0] == "termination":
            return f"Episode_Termination/{path[1]}"
        if path[0] == "curriculum":
            return f"Curriculum/{'/'.join(path[1:])}"
        if path[0] == "metrics":
            return f"Metrics/{'/'.join(path[1:])}"
        return f"Episode/{'/'.join(path)}"

    def _handle_discount(self, discount: torch.Tensor) -> None:
        if self.discount_mode == "ignore" or self._discount_warning_emitted:
            return
        if torch.allclose(discount, torch.ones_like(discount)):
            return
        message = (
            "AA emitted a non-unit per-step discount, but current Z-RL PPO does not consume it. "
            "Training will use Z-RL's standard done-based return semantics."
        )
        if self.discount_mode == "error":
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        self._discount_warning_emitted = True

    def _validate_observations(self, tensordict: TensorDictBase) -> None:
        missing = [key for key in self.observation_keys if key not in tensordict]
        if missing:
            raise KeyError(
                f"AA observation groups {missing} were not found. Available observations: {list(tensordict.keys())}"
            )

    def _validate_actions(self, actions: torch.Tensor) -> None:
        expected = (self.num_envs, self.num_actions)
        if tuple(actions.shape) != expected:
            raise ValueError(
                f"Expected actions with shape {expected}, got {tuple(actions.shape)}"
            )
        if actions.device != torch.device(self.device):
            raise ValueError(
                f"Expected actions on AA device {self.device}, got {actions.device}. "
                "The Z-RL runner should move actions to env.device before env.step()."
            )

    def _resolve_max_episode_length(self, value: int | torch.Tensor) -> int:
        if isinstance(value, int):
            return value
        flat = value.reshape(-1)
        if flat.numel() == 0 or not torch.equal(flat, flat[0].expand_as(flat)):
            raise ValueError(
                "Current Z-RL runner requires a uniform AA max_episode_length"
            )
        return int(flat[0].item())
