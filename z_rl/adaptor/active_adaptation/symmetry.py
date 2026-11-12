# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compile Active Adaptation symmetry transforms for Z-RL."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any

from z_rl.extensions import TensorMirrorSpec, augment_mirrored_tensor


class ActiveAdaptationSymmetryCompiler:
    """Cache AA observation-group and action transforms as signed permutations."""

    def __init__(self, env: Any) -> None:
        """Compile transforms from an initialized AA vector-environment wrapper."""
        self.env = env
        self.base_env = env.unwrapped
        self.obs_specs = self._compile_observation_specs()
        self.action_spec = self._compile_action_spec()

    @torch.no_grad()
    def augment(
        self,
        obs: TensorDict | None = None,
        actions: torch.Tensor | None = None,
    ) -> tuple[TensorDict | None, torch.Tensor | None]:
        """Append one mirrored copy of any provided observations and actions."""
        obs_aug = None
        if obs is not None:
            augmented = {}
            for group_name, group_obs in obs.items():
                if not torch.is_tensor(group_obs):
                    raise TypeError(
                        f"Observation group '{group_name}' must be a dense tensor for symmetry augmentation. "
                        "Functional observation groups are not supported yet."
                    )
                try:
                    spec = self.obs_specs[group_name]
                except KeyError as exc:
                    raise KeyError(
                        f"Observation group '{group_name}' was not selected when AA symmetry was compiled."
                    ) from exc
                augmented[group_name] = augment_mirrored_tensor(group_obs, spec)
            obs_aug = TensorDict(
                augmented,
                batch_size=[obs.batch_size[0] * 2],
                device=obs.device,
            )

        actions_aug = None
        if actions is not None:
            actions_aug = augment_mirrored_tensor(actions, self.action_spec)
        return obs_aug, actions_aug

    def _compile_observation_specs(self) -> dict[str, TensorMirrorSpec]:
        observation_groups = self.base_env.observation_groups
        current_observations = self.env.get_observations()
        specs = {}
        for group_name in self.env.observation_keys:
            try:
                group = observation_groups[group_name]
            except KeyError as exc:
                raise KeyError(
                    f"AA observation group '{group_name}' is unavailable for symmetry compilation."
                ) from exc

            value = current_observations.get(group_name)
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Observation group '{group_name}' must be a dense tensor for symmetry augmentation. "
                    "Functional observation groups are not supported yet."
                )
            if value.ndim != 2:
                raise ValueError(
                    f"Observation group '{group_name}' must have shape (num_envs, features) for symmetry "
                    f"augmentation, got {tuple(value.shape)}."
                )

            transform = group.symmetry_transform()
            spec = self._from_aa_transform(transform, name=f"observation/{group_name}")
            if len(spec) != value.shape[-1]:
                raise ValueError(
                    f"Observation group '{group_name}' symmetry width {len(spec)} does not match "
                    f"observation width {value.shape[-1]}."
                )
            specs[group_name] = spec.to(self.env.device)
        return specs

    def _compile_action_spec(self) -> TensorMirrorSpec:
        action_manager = self.base_env.input_managers[self.env.action_key]
        transform = action_manager.symmetry_transform()
        spec = self._from_aa_transform(transform, name=f"action/{self.env.action_key}")
        if len(spec) != self.env.num_actions:
            raise ValueError(
                f"Action input '{self.env.action_key}' symmetry width {len(spec)} does not match "
                f"action width {self.env.num_actions}."
            )
        return spec.to(self.env.device)

    @staticmethod
    def _from_aa_transform(transform: Any, *, name: str) -> TensorMirrorSpec:
        if not hasattr(transform, "perm") or not hasattr(transform, "signs"):
            raise ValueError(
                f"AA {name} must provide a SymmetryTransform with 'perm' and 'signs' tensors."
            )
        if getattr(transform, "channel_signs", None) is not None:
            raise ValueError(
                f"AA {name} uses channel-wise image symmetry, which the dense-vector adaptor does not support yet."
            )
        return TensorMirrorSpec(transform.perm, transform.signs, name=name)


__all__ = ["ActiveAdaptationSymmetryCompiler"]
