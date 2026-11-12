# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compile declarative IsaacLab symmetry metadata for z-rl."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from z_rl.extensions import (
    TensorMirrorSpec,
    augment_mirrored_tensor,
    build_obs_group_mirror_spec,
)


class IsaacLabSymmetryCompiler:
    """Compile runtime-resolved term symmetry into cached tensor operations."""

    def __init__(self, env) -> None:
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
        obs_aug = None
        if obs is not None:
            augmented = {}
            for group_name, group_obs in obs.items():
                augmented[group_name] = augment_mirrored_tensor(group_obs, self.obs_specs[group_name])
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
        observation_manager = self.base_env.observation_manager
        group_specs = {}
        for group_name in observation_manager.active_terms:
            term_specs = {}
            term_names = observation_manager.active_terms[group_name]
            term_cfgs = observation_manager._group_obs_term_cfgs[group_name]
            for term_name, term_cfg in zip(term_names, term_cfgs):
                provider = getattr(term_cfg, "symmetry_transform", None)
                if not callable(provider):
                    raise ValueError(
                        f"Observation term '{group_name}/{term_name}' must declare a callable symmetry_transform."
                    )
                frame_shape = self.env.obs_format[group_name][term_name][1:]
                transform = provider(
                    env=self.base_env,
                    term_cfg=term_cfg,
                    term_shape=frame_shape,
                    term=None,
                )
                spec = TensorMirrorSpec(
                    transform.index,
                    transform.sign,
                    name=f"{group_name}/{term_name}",
                )
                self._validate_observation_scale(term_cfg, spec, group_name, term_name)
                term_specs[term_name] = spec
            group_specs[group_name] = build_obs_group_mirror_spec(
                self.env.obs_format,
                group_name,
                term_specs,
                layout_mode=self.env.obs_group_layout_mode_map[group_name],
            ).to(self.env.device)
        return group_specs

    def _compile_action_spec(self) -> TensorMirrorSpec:
        action_manager = self.base_env.action_manager
        term_specs = []
        for term_name in action_manager.active_terms:
            term = action_manager.get_term(term_name)
            provider = getattr(term.cfg, "symmetry_transform", None)
            if not callable(provider):
                raise ValueError(f"Action term '{term_name}' must declare a callable symmetry_transform.")
            transform = provider(
                env=self.base_env,
                term_cfg=term.cfg,
                term_shape=(term.action_dim,),
                term=term,
            )
            spec = TensorMirrorSpec(transform.index, transform.sign, name=f"action/{term_name}")
            if len(spec) != term.action_dim:
                raise ValueError(
                    f"Action term '{term_name}' symmetry width {len(spec)} "
                    f"does not match action width {term.action_dim}."
                )
            term_specs.append(spec)
        return TensorMirrorSpec.cat(term_specs, name="action").to(self.env.device)

    @staticmethod
    def _validate_observation_scale(term_cfg, spec: TensorMirrorSpec, group_name: str, term_name: str) -> None:
        scale = term_cfg.scale
        if scale is None or isinstance(scale, (int, float)):
            return
        scale_tensor = torch.as_tensor(scale).flatten().detach().cpu()
        if scale_tensor.numel() not in (1, len(spec)):
            raise ValueError(
                f"Observation term '{group_name}/{term_name}' scale width {scale_tensor.numel()} "
                f"does not match symmetry width {len(spec)}."
            )
        if scale_tensor.numel() > 1 and not torch.allclose(scale_tensor, scale_tensor[spec.index.cpu()]):
            raise ValueError(f"Observation term '{group_name}/{term_name}' scale is not mirror-invariant.")
