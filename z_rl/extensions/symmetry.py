# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Mapping, Sequence
from math import prod

from z_rl.env import VecEnv
from z_rl.models import MLPModel
from z_rl.storage import RolloutStorage


class TensorMirrorSpec(nn.Module):
    """Compiled signed permutation for a flat tensor's last dimension."""

    def __init__(
        self,
        index: Sequence[int] | torch.Tensor,
        sign: Sequence[float] | torch.Tensor | None = None,
        *,
        name: str = "tensor",
    ) -> None:
        super().__init__()
        index_tensor = torch.as_tensor(index, dtype=torch.long).flatten().detach().cpu()
        sign_tensor = (
            torch.ones_like(index_tensor, dtype=torch.float32)
            if sign is None
            else torch.as_tensor(sign, dtype=torch.float32).flatten().detach().cpu()
        )
        self.name = name
        self._validate(index_tensor, sign_tensor)
        self.register_buffer("index", index_tensor)
        self.register_buffer("sign", sign_tensor)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(-1, self.index) * self.sign

    def __len__(self) -> int:
        return self.index.numel()

    def repeat(self, count: int, *, name: str | None = None) -> TensorMirrorSpec:
        return TensorMirrorSpec.cat([self] * count, name=name or f"{self.name} x {count}")

    @staticmethod
    def cat(specs: Sequence[TensorMirrorSpec], *, name: str = "concatenated") -> TensorMirrorSpec:
        indices = []
        signs = []
        offset = 0
        for spec in specs:
            indices.append(spec.index.cpu() + offset)
            signs.append(spec.sign.cpu())
            offset += len(spec)
        return TensorMirrorSpec(torch.cat(indices), torch.cat(signs), name=name)

    def _validate(self, index: torch.Tensor, sign: torch.Tensor) -> None:
        if index.shape != sign.shape:
            raise ValueError(
                f"Mirror spec '{self.name}' index/sign must have equal length; got {index.numel()} and {sign.numel()}."
            )
        expected = torch.arange(index.numel())
        if not torch.equal(torch.sort(index).values, expected):
            raise ValueError(f"Mirror spec '{self.name}' index must be a permutation of [0, {index.numel()}).")
        # Reflection is an involution: applying both the permutation and its signs twice must recover the input.
        if not torch.equal(index[index], expected):
            raise ValueError(f"Mirror spec '{self.name}' permutation is not an involution.")
        if not torch.allclose(sign * sign[index], torch.ones_like(sign)):
            raise ValueError(f"Mirror spec '{self.name}' signs do not form an involution.")


def augment_mirrored_tensor(tensor: torch.Tensor, spec: TensorMirrorSpec) -> torch.Tensor:
    """Append one mirrored copy of ``tensor`` along the batch dimension."""
    return torch.cat((tensor, spec(tensor)), dim=0)


def build_obs_group_mirror_spec(
    obs_format: Mapping[str, Mapping[str, tuple[int, ...]]],
    group_name: str,
    term_specs: Mapping[str, TensorMirrorSpec],
    *,
    layout_mode: str,
) -> TensorMirrorSpec:
    """Expand per-frame term specs into a flattened observation-group spec."""
    term_formats = obs_format[group_name]
    missing = set(term_formats) - set(term_specs)
    if missing:
        raise ValueError(f"Observation group '{group_name}' is missing symmetry for terms: {sorted(missing)}.")

    resolved_specs = []
    histories = []
    for term_name, term_format in term_formats.items():
        history_length = max(int(term_format[0]), 1)
        frame_dim = prod(term_format[1:])
        term_spec = term_specs[term_name]
        if len(term_spec) != frame_dim:
            raise ValueError(
                f"Observation term '{group_name}/{term_name}' symmetry width {len(term_spec)} "
                f"does not match frame width {frame_dim}."
            )
        resolved_specs.append(term_spec)
        histories.append(history_length)

    if layout_mode == "term_major":
        # Each term stores all of its history frames contiguously before the next term.
        expanded = [spec.repeat(history) for spec, history in zip(resolved_specs, histories)]
        return TensorMirrorSpec.cat(expanded, name=group_name)
    if layout_mode == "history_major":
        if len(set(histories)) != 1:
            raise ValueError(
                f"History-major observation group '{group_name}' requires a common history length; got {histories}."
            )
        # A full frame is concatenated first, then repeated for every history step.
        frame_spec = TensorMirrorSpec.cat(resolved_specs, name=f"{group_name}:frame")
        return frame_spec.repeat(histories[0], name=group_name)
    raise ValueError(f"Unsupported observation layout mode '{layout_mode}'.")


class Symmetry:
    """Symmetry augmentation and mirror-loss helper for PPO."""

    def __init__(
        self,
        env: VecEnv,
        batch_is_augmented: bool = False,
        use_mirror_loss: bool = False,
        log_mirror_loss: bool = False,
        mirror_loss_log_interval: int = 100,
        mirror_loss_coeff: float = 0.0,
    ) -> None:
        """Resolve and store symmetry configuration."""
        if log_mirror_loss and mirror_loss_log_interval < 1:
            raise ValueError("mirror_loss_log_interval must be positive.")
        self.env = env
        # PPO resolves this once: mirror loss can reuse an augmented mini-batch instead of transforming it again.
        self._batch_is_augmented = batch_is_augmented
        self.use_mirror_loss = use_mirror_loss
        self.log_mirror_loss = log_mirror_loss
        self.mirror_loss_log_interval = mirror_loss_log_interval
        self.mirror_loss_coeff = mirror_loss_coeff
        env.compile_symmetry()

    def augment_batch(self, batch: RolloutStorage.Batch, original_batch_size: int) -> None:
        """Append mirrored samples to a rollout mini-batch without expanding its storage."""
        batch.observations, batch.actions = self.env.augment_symmetry(
            obs=batch.observations,
            actions=batch.actions,
        )
        augmentation_count = int(batch.observations.batch_size[0] / original_batch_size)
        # Mirrored samples share scalar rollout targets with their original samples.
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(augmentation_count, 1)
        batch.values = batch.values.repeat(augmentation_count, 1)
        batch.advantages = batch.advantages.repeat(augmentation_count, 1)
        batch.returns = batch.returns.repeat(augmentation_count, 1)

    def compute_loss(self, actor: MLPModel, batch: RolloutStorage.Batch, original_batch_size: int) -> torch.Tensor:
        """Compute the differentiable mirror loss used for optimization."""
        return self._compute_mirror_loss(actor, batch, original_batch_size)

    @torch.no_grad()
    def compute_metric(self, actor: MLPModel, batch: RolloutStorage.Batch, original_batch_size: int) -> torch.Tensor:
        """Compute a detached mirror metric for logging."""
        return self._compute_mirror_loss(actor, batch, original_batch_size)

    def _compute_mirror_loss(
        self, actor: MLPModel, batch: RolloutStorage.Batch, original_batch_size: int
    ) -> torch.Tensor:
        """Compute mirror consistency for one augmented mini-batch."""
        observations = batch.observations
        if not self._batch_is_augmented:
            observations, _ = self.env.augment_symmetry(obs=observations, actions=None)

        policy_mean = actor(observations)
        _, transformed_mean = self.env.augment_symmetry(obs=None, actions=policy_mean[:original_batch_size])

        prediction = policy_mean[original_batch_size:]
        # The mirrored original-policy output is a fixed target; gradients update only the mirrored-state prediction.
        target = transformed_mean[original_batch_size:].detach()
        return nn.functional.mse_loss(prediction, target)


def resolve_symmetry_config(alg_cfg: dict, env: VecEnv) -> dict:
    """Resolve the rollout-augmentation switch and advanced symmetry settings.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        env: Environment object.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    symmetry_augmentation = alg_cfg.get("symmetry_augmentation", False)
    symmetry_cfg = alg_cfg.get("symmetry_cfg")
    if symmetry_augmentation and symmetry_cfg is None:
        symmetry_cfg = {}

    if symmetry_cfg is not None:
        symmetry_cfg["env"] = env
    alg_cfg["symmetry_cfg"] = symmetry_cfg
    return alg_cfg
