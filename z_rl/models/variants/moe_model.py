# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from z_rl.modules import MoE

from z_rl.models.composition import ComposableModel, HeadSpec


@dataclass(slots=True)
class MoEHeadSpec(HeadSpec):
    """Explicit head spec that builds a Mixture-of-Experts output head."""

    num_experts: int = 4
    expert_hidden_dims: tuple[int, ...] | list[int] | int = (256,)
    gate_hidden_dims: tuple[int, ...] | list[int] | int | None = None

    def validate(self, model: nn.Module) -> None:
        """Validate MoE-specific parameters before the head is rebuilt."""
        if self.num_experts <= 0:
            raise ValueError(f"`num_experts` must be positive, got {self.num_experts}.")
        if isinstance(self.expert_hidden_dims, (tuple, list)) and len(self.expert_hidden_dims) == 0:
            raise ValueError("`expert_hidden_dims` can not be empty.")

    def build_head(self, model: nn.Module, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        """Build the MoE head for the provided model dimensions."""
        return MoE(
            input_dim,
            output_dim,
            self.num_experts,
            self.expert_hidden_dims,
            gate_hidden_dims=self.gate_hidden_dims,
            activation=activation,
        )


class MoEModel(ComposableModel):
    """MLPModel variant whose head is a Mixture-of-Experts MLP.

    Data flow: ``obs groups -> (per-group normalization) -> concat latent -> MoE head -> (distribution) -> output``.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_experts: int = 4,
        expert_hidden_dims: tuple[int, ...] | list[int] | int = (256,),
        gate_hidden_dims: tuple[int, ...] | list[int] | int | None = None,
        pretrained_expert_path: str | None = None,
        pretrained_expert_state_dict_key: str | None = None,
        load_pretrained_expert_strict: bool = True,
        pretrained_expert_target_indices: list[int] | None = None,
        pretrained_expert_target_index: int | None = None,
        pretrained_expert_specs: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the MoE-based model with an explicit MoE head spec."""
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            head_spec=MoEHeadSpec(
                num_experts=num_experts,
                expert_hidden_dims=expert_hidden_dims,
                gate_hidden_dims=gate_hidden_dims,
            ),
        )

        if pretrained_expert_target_index is not None:
            if pretrained_expert_target_indices is not None:
                raise ValueError("Use only one of `pretrained_expert_target_index` or `pretrained_expert_target_indices`.")
            pretrained_expert_target_indices = [pretrained_expert_target_index]

        if pretrained_expert_specs is not None:
            if pretrained_expert_path is not None:
                raise ValueError("Use `pretrained_expert_specs` or `pretrained_expert_path`, not both.")
            self._load_pretrained_expert_specs(pretrained_expert_specs, default_strict=load_pretrained_expert_strict)
        elif pretrained_expert_path is not None:
            self.load_pretrained_experts(
                pretrained_expert_path,
                state_dict_key=pretrained_expert_state_dict_key,
                strict=load_pretrained_expert_strict,
                target_expert_indices=pretrained_expert_target_indices,
            )

    def _load_pretrained_expert_specs(self, specs: list[dict[str, Any]], default_strict: bool) -> None:
        """Load multiple pretrained expert sources from per-source specs."""
        for spec_idx, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise TypeError(f"`pretrained_expert_specs[{spec_idx}]` must be a dict, got {type(spec).__name__}.")

            path = spec.get("path")
            if not isinstance(path, str) or len(path) == 0:
                raise ValueError(f"`pretrained_expert_specs[{spec_idx}].path` must be a non-empty string.")

            state_dict_key = spec.get("state_dict_key")
            if state_dict_key is not None and not isinstance(state_dict_key, str):
                raise TypeError(
                    f"`pretrained_expert_specs[{spec_idx}].state_dict_key` must be str or None, "
                    f"got {type(state_dict_key).__name__}."
                )

            strict = spec.get("strict", default_strict)
            if not isinstance(strict, bool):
                raise TypeError(
                    f"`pretrained_expert_specs[{spec_idx}].strict` must be bool, got {type(strict).__name__}."
                )

            single_index = spec.get("target_expert_index")
            indices = spec.get("target_expert_indices")
            if single_index is not None and indices is not None:
                raise ValueError(
                    f"`pretrained_expert_specs[{spec_idx}]` can only provide one of "
                    "`target_expert_index` and `target_expert_indices`."
                )
            if single_index is not None:
                if not isinstance(single_index, int):
                    raise TypeError(
                        f"`pretrained_expert_specs[{spec_idx}].target_expert_index` must be int, "
                        f"got {type(single_index).__name__}."
                    )
                indices = [single_index]

            self.load_pretrained_experts(
                path,
                state_dict_key=state_dict_key,
                strict=strict,
                target_expert_indices=indices,
            )

    def load_pretrained_experts(
        self,
        path: str,
        state_dict_key: str | None = None,
        strict: bool = True,
        target_expert_indices: list[int] | None = None,
    ) -> None:
        """Load expert parameters into ``self.head.experts`` from one checkpoint.

        Supports two source layouts:
        1) MoE expert checkpoints (keys like ``head.experts.weights.0`` or ``weights.0``).
        2) MLPModel checkpoints (keys like ``head.0.weight``), mapped to MoE experts.
        """
        if not hasattr(self.head, "experts"):
            raise TypeError(f"`{type(self.head).__name__}` has no `experts` module, can not load pretrained experts.")

        target_indices = self._normalize_target_indices(target_expert_indices)
        loaded = torch.load(path, weights_only=False, map_location="cpu")
        source_state_dict = self._resolve_source_state_dict(loaded, state_dict_key)
        expert_state_dict = self._extract_expert_state_dict(source_state_dict, target_indices=target_indices)
        self.head.experts.load_state_dict(expert_state_dict, strict=strict)

    def _normalize_target_indices(self, target_expert_indices: list[int] | None) -> list[int] | None:
        """Validate and normalize target expert indices."""
        if target_expert_indices is None:
            return None

        if len(target_expert_indices) == 0:
            raise ValueError("`target_expert_indices` can not be empty.")

        normalized = sorted(set(target_expert_indices))
        num_experts = self.head.experts.num_experts
        for idx in normalized:
            if idx < 0 or idx >= num_experts:
                raise ValueError(f"Expert index out of range: {idx}. num_experts={num_experts}.")
        return normalized

    def _resolve_source_state_dict(self, loaded: Any, state_dict_key: str | None) -> dict[str, torch.Tensor]:
        """Resolve the state dict object from common checkpoint layouts."""
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected checkpoint dict, got {type(loaded).__name__}.")

        if state_dict_key is not None:
            if state_dict_key not in loaded:
                raise KeyError(f"Key `{state_dict_key}` not found in checkpoint.")
            state_dict = loaded[state_dict_key]
            if not isinstance(state_dict, dict):
                raise TypeError(f"Checkpoint key `{state_dict_key}` is not a state dict.")
            return state_dict

        if "actor_state_dict" in loaded and isinstance(loaded["actor_state_dict"], dict):
            return loaded["actor_state_dict"]
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            return loaded["state_dict"]
        return loaded

    def _extract_expert_state_dict(
        self,
        source_state_dict: dict[str, torch.Tensor],
        target_indices: list[int] | None,
    ) -> dict[str, torch.Tensor]:
        """Extract expert keys for ``self.head.experts`` from a potentially larger state dict."""
        target_state = self.head.experts.state_dict()
        matched: dict[str, torch.Tensor] = {}

        for source_key, tensor in source_state_dict.items():
            for target_key in target_state.keys():
                if source_key == target_key or source_key.endswith(f".{target_key}"):
                    matched[target_key] = tensor
                    break

        if matched:
            return self._adapt_expert_tensor_shapes(matched, target_state, target_indices)

        mapped = self._map_mlp_head_to_experts(source_state_dict, target_state=target_state, target_indices=target_indices)
        if mapped is not None:
            return mapped

        expected = ", ".join(tuple(target_state.keys())[:2])
        raise KeyError(
            "No expert parameters found in checkpoint. "
            f"Expected keys ending with `head.experts.<...>` / `<...>.{expected}` "
            "or an MLP head state_dict like `head.0.weight`."
        )

    def _adapt_expert_tensor_shapes(
        self,
        matched: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        target_indices: list[int] | None,
    ) -> dict[str, torch.Tensor]:
        """Adapt source expert tensors to target shapes with optional selective expert loading."""
        adapted: dict[str, torch.Tensor] = {}

        for key, source_tensor in matched.items():
            target_tensor = target_state[key].clone()
            source_tensor = source_tensor.to(target_tensor.device, dtype=target_tensor.dtype)

            if source_tensor.shape == target_tensor.shape:
                if target_indices is None:
                    adapted[key] = source_tensor
                else:
                    target_tensor[target_indices] = source_tensor[target_indices]
                    adapted[key] = target_tensor
                continue

            if source_tensor.shape == target_tensor.shape[1:]:
                if target_indices is None:
                    target_tensor[:] = source_tensor.unsqueeze(0).expand_as(target_tensor)
                else:
                    target_tensor[target_indices] = source_tensor.unsqueeze(0).expand(
                        len(target_indices), *source_tensor.shape
                    )
                adapted[key] = target_tensor
                continue

            raise ValueError(
                f"Shape mismatch for expert key `{key}`. "
                f"Expected {tuple(target_tensor.shape)} or {tuple(target_tensor.shape[1:])}, "
                f"got {tuple(source_tensor.shape)}."
            )

        return adapted

    def _map_mlp_head_to_experts(
        self,
        source_state_dict: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        target_indices: list[int] | None,
    ) -> dict[str, torch.Tensor] | None:
        """Map MLP head linear layers to MoE expert parameters.

        MLP linear weight shape is ``[out_dim, in_dim]``; MoE expert weight expects ``[E, in_dim, out_dim]``.
        """
        pattern = re.compile(r"^head\.(\d+)\.(weight|bias)$")
        linear_by_layer: dict[int, dict[str, torch.Tensor]] = {}

        for key, tensor in source_state_dict.items():
            match = pattern.match(key)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            param_kind = match.group(2)
            linear_by_layer.setdefault(layer_idx, {})[param_kind] = tensor

        if not linear_by_layer:
            return None

        source_linears: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in sorted(linear_by_layer.keys()):
            params = linear_by_layer[layer_idx]
            if "weight" not in params or "bias" not in params:
                continue
            weight, bias = params["weight"], params["bias"]
            if weight.dim() != 2 or bias.dim() != 1:
                continue
            source_linears.append((weight, bias))

        num_layers = len(self.head.experts.weights)
        if len(source_linears) != num_layers:
            return None

        all_indices = list(range(self.head.experts.num_experts))
        selected_indices = all_indices if target_indices is None else target_indices

        mapped: dict[str, torch.Tensor] = {}
        for layer_idx, (src_weight, src_bias) in enumerate(source_linears):
            target_weight = target_state[f"weights.{layer_idx}"].clone()
            target_bias = target_state[f"biases.{layer_idx}"].clone()

            expected_in, expected_out = target_weight.shape[1], target_weight.shape[2]
            if src_weight.shape != (expected_out, expected_in):
                raise ValueError(
                    f"MLP layer {layer_idx} shape mismatch. "
                    f"Expected source weight {(expected_out, expected_in)}, got {tuple(src_weight.shape)}."
                )
            if src_bias.shape != (expected_out,):
                raise ValueError(
                    f"MLP layer {layer_idx} bias mismatch. "
                    f"Expected source bias {(expected_out,)}, got {tuple(src_bias.shape)}."
                )

            converted_weight = src_weight.transpose(0, 1).to(target_weight.device, dtype=target_weight.dtype)
            converted_bias = src_bias.to(target_bias.device, dtype=target_bias.dtype)

            target_weight[selected_indices] = converted_weight.unsqueeze(0).expand(len(selected_indices), -1, -1)
            target_bias[selected_indices] = converted_bias.unsqueeze(0).expand(len(selected_indices), -1)

            mapped[f"weights.{layer_idx}"] = target_weight
            mapped[f"biases.{layer_idx}"] = target_bias

        return mapped
