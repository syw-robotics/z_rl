# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from z_rl.env import VecEnv
from z_rl.storage import RolloutStorage
from z_rl.utils import bind_matching_fields, resolve_spec, spec_init_field_names

from ..ppo import PPO
from .specs import PPOLossSpec

SpecRef = PPOLossSpec | dict[str, Any] | str | type | None


def _install_loss_spec_coefs(algo: object, spec: object) -> None:
    """Copy ``*_coef`` fields from the spec onto the algorithm for ``PPO.update()``."""
    for name in spec_init_field_names(spec):
        if name.endswith("_coef"):
            setattr(algo, name, getattr(spec, name))


class ComposablePPO(PPO):
    """PPO variant that explicitly applies one optional loss spec after base loss computation.

    Named subclasses can set ``loss_spec_class`` so ``class_name`` still works without a nested
    ``loss_spec`` config. Users otherwise point ``algorithm.loss_spec`` at a spec class or config dict.
    """

    loss_spec_class: SpecRef = None

    def __init__(self, *args, loss_spec: SpecRef = None, **kwargs) -> None:
        """Initialize PPO and store the additional loss spec."""
        super().__init__(*args, **kwargs)
        if loss_spec is None:
            loss_spec = type(self).loss_spec_class
        self.loss_spec = resolve_spec(loss_spec)
        if self.loss_spec is None:
            raise ValueError("`ComposablePPO` requires a `loss_spec` to be provided for additional loss computation.")
        self.loss_spec.validate(self)
        _install_loss_spec_coefs(self, self.loss_spec)

    def compute_loss(self, minibatch: RolloutStorage.Batch) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute base PPO loss and merge any additional losses from the configured spec."""
        opt_losses, non_opt_losses = super().compute_loss(minibatch)
        extra_opt_losses, extra_non_opt_losses = self.loss_spec.compute(self, minibatch)
        opt_losses.update(extra_opt_losses)
        non_opt_losses.update(extra_non_opt_losses)
        return opt_losses, non_opt_losses

    def act(self, obs: TensorDict) -> torch.Tensor:
        """ Subclasses can override this method when a PPO variant needs full control over rollout-time action generation
        and transition bookkeeping.
        """
        return super().act(obs)

    def forward_actor_for_update(self, minibatch: RolloutStorage.Batch) -> None:
        """Subclasses can override update-time actor forwards."""
        super().forward_actor_for_update(minibatch)

    @classmethod
    def build_loss_spec(cls, env: VecEnv, algorithm_cfg: dict) -> PPOLossSpec:
        """Build the loss spec from ``algorithm_cfg["loss_spec"]`` or ``cls.loss_spec_class``.

        Leftover algorithm-config keys that match spec init fields (including ``*_coef``) are bound
        onto the spec so named IsaacLab presets can keep flat coefficient fields.
        """
        spec_ref = algorithm_cfg.pop("loss_spec", cls.loss_spec_class)
        if spec_ref is None:
            raise ValueError("`ComposablePPO` requires a `loss_spec` to be provided for additional loss computation.")
        spec = resolve_spec(spec_ref)
        bind_matching_fields(spec, algorithm_cfg)
        return spec

    @classmethod
    def _build_algorithm_extra_kwargs(cls, env: VecEnv, algorithm_cfg: dict) -> dict:
        """Build composable PPO-specific keyword arguments for shared PPO construction."""
        return {"loss_spec": cls.build_loss_spec(env, algorithm_cfg)}
