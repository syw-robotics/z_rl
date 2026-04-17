# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from tensordict import TensorDict

from z_rl.env import VecEnv
from z_rl.extensions import resolve_symmetry_config
from z_rl.models import MLPModel
from z_rl.storage import RolloutStorage
from z_rl.utils import inject_obs_time_slice_map, resolve_callable, resolve_obs_groups

from ..ppo import PPO
from .specs import PPOLossSpec


class ComposablePPO(PPO):
    """PPO variant that explicitly applies one optional loss spec after base loss computation."""

    def __init__(self, *args, loss_spec: PPOLossSpec | None = None, **kwargs) -> None:
        """Initialize PPO and store the optional additional loss spec."""
        super().__init__(*args, **kwargs)
        self.loss_spec = loss_spec
        if self.loss_spec is not None:
            self.loss_spec.validate(self)
        else:
            raise ValueError("`ComposablePPO` requires a `loss_spec` to be provided for additional loss computation.")

    def compute_loss(self, minibatch: RolloutStorage.Batch) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute base PPO loss and merge any additional losses from the configured spec."""
        opt_losses, non_opt_losses = super().compute_loss(minibatch)
        if self.loss_spec is not None:
            extra_opt_losses, extra_non_opt_losses = self.loss_spec.compute(self, minibatch)
            opt_losses.update(extra_opt_losses)
            non_opt_losses.update(extra_non_opt_losses)
        return opt_losses, non_opt_losses

    def act(self, obs: TensorDict) -> torch.Tensor:
        """ Subclasses can override this method when a PPO variant needs full control over rollout-time action generation
        and transition bookkeeping.
        """
        return super().act(obs)

    @classmethod
    def build_loss_spec(cls, env: VecEnv, algorithm_cfg: dict) -> PPOLossSpec:
        """Build the loss spec for this PPO variant from the environment and algorithm config."""
        raise NotImplementedError(f"`{cls.__name__}` must override `build_loss_spec()`.")

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "ComposablePPO":
        """Construct a composable PPO variant using the shared PPO assembly flow."""
        cfg["algorithm"].pop("class_name", None)
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        inject_obs_time_slice_map(cfg["actor"], actor_class, env)
        inject_obs_time_slice_map(cfg["critic"], critic_class, env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        loss_spec = cls.build_loss_spec(env, cfg["algorithm"])

        alg: ComposablePPO = cls(
            actor,
            critic,
            storage,
            loss_spec=loss_spec,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        return alg
