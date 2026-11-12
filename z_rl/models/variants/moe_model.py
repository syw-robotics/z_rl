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

from z_rl.modules import MoE

from z_rl.models.composition import ComposableModel, HeadSpec

_MLP_HEAD_LINEAR_KEY = re.compile(r"^head\.(\d+)\.(weight|bias)$")
_DEFAULT_STATE_DICT_KEYS = ("actor_state_dict", "state_dict")


@dataclass
class MoEHeadSpec(HeadSpec):
    """Explicit head spec that builds a Mixture-of-Experts output head."""

    num_experts: int = 4
    expert_hidden_dims: tuple[int, ...] | list[int] = (256,)
    gate_hidden_dims: tuple[int, ...] | list[int] | None = None
    top_k: int | None = None

    def validate(self, model: nn.Module) -> None:
        """Validate MoE-specific parameters before the head is rebuilt."""
        if self.num_experts <= 0:
            raise ValueError(f"`num_experts` must be positive, got {self.num_experts}.")
        if len(self.expert_hidden_dims) == 0:
            raise ValueError("`expert_hidden_dims` can not be empty.")

    def build(self, model: nn.Module, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        """Build the MoE head for the provided model dimensions."""
        return MoE(
            input_dim,
            output_dim,
            self.num_experts,
            self.expert_hidden_dims,
            gate_hidden_dims=self.gate_hidden_dims,
            activation=activation,
            top_k=self.top_k,
        )


class MoEModel(ComposableModel):
    """Named MLP preset whose head is a Mixture-of-Experts MLP.

    Data flow: ``obs groups -> (per-group normalization) -> concat latent -> MoE head -> (distribution) -> output``.
    Extra ``__init__`` is only for pretrained expert loading; routing hyperparameters bind onto ``MoEHeadSpec``.
    """

    head_spec_class = MoEHeadSpec

    def __init__(
        self,
        *args,
        pretrained_expert_path: str | None = None,
        pretrained_expert_state_dict_key: str | None = None,
        load_pretrained_expert_strict: bool = True,
        pretrained_expert_target_indices: list[int] | None = None,
        pretrained_expert_target_index: int | None = None,
        pretrained_expert_specs: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the MoE model, then optionally load pretrained expert weights."""
        super().__init__(*args, **kwargs)
        for load in _iter_pretrained_expert_loads(
            path=pretrained_expert_path,
            state_dict_key=pretrained_expert_state_dict_key,
            strict=load_pretrained_expert_strict,
            target_indices=pretrained_expert_target_indices,
            target_index=pretrained_expert_target_index,
            specs=pretrained_expert_specs,
        ):
            self.load_pretrained_experts(
                load.path,
                state_dict_key=load.state_dict_key,
                strict=load.strict,
                target_expert_indices=load.target_expert_indices,
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
        experts = _require_experts(self.head)
        target_indices = _normalize_target_indices(target_expert_indices, experts.num_experts)
        source_state_dict = _load_source_state_dict(path, state_dict_key)
        expert_state_dict = _to_expert_state_dict(source_state_dict, experts, target_indices)
        experts.load_state_dict(expert_state_dict, strict=strict)


"""
Pretrained expert loading
"""


@dataclass(frozen=True)
class _PretrainedExpertLoad:
    """One checkpoint source used to initialize ``head.experts``."""

    path: str
    state_dict_key: str | None = None
    strict: bool = True
    target_expert_indices: list[int] | None = None


def _iter_pretrained_expert_loads(
    path: str | None,
    state_dict_key: str | None,
    strict: bool,
    target_indices: list[int] | None,
    target_index: int | None,
    specs: list[dict[str, Any]] | None,
) -> list[_PretrainedExpertLoad]:
    """Normalize constructor kwargs into a list of expert-load requests."""
    if target_index is not None:
        if target_indices is not None:
            raise ValueError("Use only one of `pretrained_expert_target_index` or `pretrained_expert_target_indices`.")
        target_indices = [target_index]

    if specs is not None:
        if path is not None:
            raise ValueError("Use `pretrained_expert_specs` or `pretrained_expert_path`, not both.")
        return [
            _parse_pretrained_expert_spec(spec, spec_idx, default_strict=strict)
            for spec_idx, spec in enumerate(specs)
        ]

    if path is None:
        return []
    return [
        _PretrainedExpertLoad(
            path=path,
            state_dict_key=state_dict_key,
            strict=strict,
            target_expert_indices=target_indices,
        )
    ]


def _parse_pretrained_expert_spec(spec: Any, spec_idx: int, default_strict: bool) -> _PretrainedExpertLoad:
    """Validate one ``pretrained_expert_specs`` item and convert it to a load request."""
    prefix = f"pretrained_expert_specs[{spec_idx}]"
    if not isinstance(spec, dict):
        raise TypeError(f"`{prefix}` must be a dict, got {type(spec).__name__}.")

    path = spec.get("path")
    if not isinstance(path, str) or len(path) == 0:
        raise ValueError(f"`{prefix}.path` must be a non-empty string.")

    state_dict_key = spec.get("state_dict_key")
    if state_dict_key is not None and not isinstance(state_dict_key, str):
        raise TypeError(f"`{prefix}.state_dict_key` must be str or None, got {type(state_dict_key).__name__}.")

    load_strict = spec.get("strict", default_strict)
    if not isinstance(load_strict, bool):
        raise TypeError(f"`{prefix}.strict` must be bool, got {type(load_strict).__name__}.")

    single_index = spec.get("target_expert_index")
    indices = spec.get("target_expert_indices")
    if single_index is not None and indices is not None:
        raise ValueError(f"`{prefix}` can only provide one of `target_expert_index` and `target_expert_indices`.")
    if single_index is not None:
        if not isinstance(single_index, int):
            raise TypeError(f"`{prefix}.target_expert_index` must be int, got {type(single_index).__name__}.")
        indices = [single_index]

    return _PretrainedExpertLoad(
        path=path,
        state_dict_key=state_dict_key,
        strict=load_strict,
        target_expert_indices=indices,
    )


def _require_experts(head: nn.Module) -> nn.Module:
    """Return ``head.experts`` or raise if the installed head is not MoE."""
    if not hasattr(head, "experts"):
        raise TypeError(f"`{type(head).__name__}` has no `experts` module, can not load pretrained experts.")
    return head.experts


def _normalize_target_indices(target_expert_indices: list[int] | None, num_experts: int) -> list[int] | None:
    """Validate, deduplicate, and sort target expert indices."""
    if target_expert_indices is None:
        return None
    if len(target_expert_indices) == 0:
        raise ValueError("`target_expert_indices` can not be empty.")

    normalized = sorted(set(target_expert_indices))
    for idx in normalized:
        if idx < 0 or idx >= num_experts:
            raise ValueError(f"Expert index out of range: {idx}. num_experts={num_experts}.")
    return normalized


def _load_source_state_dict(path: str, state_dict_key: str | None) -> dict[str, torch.Tensor]:
    """Load a checkpoint file and resolve the tensor state dict inside it."""
    loaded = torch.load(path, weights_only=False, map_location="cpu")
    return _resolve_source_state_dict(loaded, state_dict_key)


def _resolve_source_state_dict(loaded: Any, state_dict_key: str | None) -> dict[str, torch.Tensor]:
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

    for key in _DEFAULT_STATE_DICT_KEYS:
        state_dict = loaded.get(key)
        if isinstance(state_dict, dict):
            return state_dict
    return loaded


def _to_expert_state_dict(
    source_state_dict: dict[str, torch.Tensor],
    experts: nn.Module,
    target_indices: list[int] | None,
) -> dict[str, torch.Tensor]:
    """Convert a checkpoint state dict into ``experts.load_state_dict`` tensors."""
    target_state = experts.state_dict()
    matched = _match_moe_expert_tensors(source_state_dict, target_state)
    if matched:
        return _adapt_expert_tensor_shapes(matched, target_state, target_indices)

    mapped = _map_mlp_head_to_experts(source_state_dict, experts, target_state, target_indices)
    if mapped is not None:
        return mapped

    expected = ", ".join(tuple(target_state.keys())[:2])
    raise KeyError(
        "No expert parameters found in checkpoint. "
        f"Expected keys ending with `head.experts.<...>` / `<...>.{expected}` "
        "or an MLP head state_dict like `head.0.weight`."
    )


def _match_moe_expert_tensors(
    source_state_dict: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Collect source tensors whose keys map onto ``experts`` parameter names."""
    target_keys = tuple(target_state)
    matched: dict[str, torch.Tensor] = {}
    for source_key, tensor in source_state_dict.items():
        target_key = _as_expert_param_key(source_key, target_keys)
        if target_key is not None:
            matched[target_key] = tensor
    return matched


def _as_expert_param_key(source_key: str, target_keys: tuple[str, ...]) -> str | None:
    """Map a checkpoint key onto an ``experts`` parameter name, if any."""
    for target_key in target_keys:
        if source_key == target_key or source_key.endswith(f".{target_key}"):
            return target_key
    return None


def _adapt_expert_tensor_shapes(
    matched: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    target_indices: list[int] | None,
) -> dict[str, torch.Tensor]:
    """Copy source expert tensors into the live expert parameter shapes."""
    return {
        key: _place_expert_tensor(source_tensor, target_state[key], target_indices, key)
        for key, source_tensor in matched.items()
    }


def _place_expert_tensor(
    source_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    target_indices: list[int] | None,
    key: str,
) -> torch.Tensor:
    """Place a source expert tensor into the corresponding target parameter.

    Accepted source shapes:
    - full stacked experts ``[E, ...]``: copy all experts, or the selected indices
    - a single expert ``[...]``: broadcast onto all experts, or the selected indices
    """
    source_tensor = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)

    if source_tensor.shape == target_tensor.shape:
        if target_indices is None:
            return source_tensor
        placed = target_tensor.clone()
        placed[target_indices] = source_tensor[target_indices]
        return placed

    if source_tensor.shape == target_tensor.shape[1:]:
        placed = target_tensor.clone()
        placed[slice(None) if target_indices is None else target_indices] = source_tensor
        return placed

    raise ValueError(
        f"Shape mismatch for expert key `{key}`. "
        f"Expected {tuple(target_tensor.shape)} or {tuple(target_tensor.shape[1:])}, "
        f"got {tuple(source_tensor.shape)}."
    )


def _map_mlp_head_to_experts(
    source_state_dict: dict[str, torch.Tensor],
    experts: nn.Module,
    target_state: dict[str, torch.Tensor],
    target_indices: list[int] | None,
) -> dict[str, torch.Tensor] | None:
    """Map MLP head linear layers to MoE expert parameters.

    MLP linear weight shape is ``[out_dim, in_dim]``; MoE expert weight expects ``[E, in_dim, out_dim]``.
    """
    source_linears = _collect_mlp_head_linears(source_state_dict)
    if source_linears is None:
        return None

    num_layers = len(experts.weights)
    if len(source_linears) != num_layers:
        return None

    dest = slice(None) if target_indices is None else target_indices
    mapped: dict[str, torch.Tensor] = {}
    for layer_idx, (src_weight, src_bias) in enumerate(source_linears):
        target_weight = target_state[f"weights.{layer_idx}"]
        target_bias = target_state[f"biases.{layer_idx}"]
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

        converted_weight = src_weight.transpose(0, 1).to(device=target_weight.device, dtype=target_weight.dtype)
        converted_bias = src_bias.to(device=target_bias.device, dtype=target_bias.dtype)
        placed_weight = target_weight.clone()
        placed_bias = target_bias.clone()
        placed_weight[dest] = converted_weight
        placed_bias[dest] = converted_bias
        mapped[f"weights.{layer_idx}"] = placed_weight
        mapped[f"biases.{layer_idx}"] = placed_bias
    return mapped


def _collect_mlp_head_linears(
    source_state_dict: dict[str, torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    """Collect ``(weight, bias)`` pairs for MLP head linear layers, in layer order.

    Sequential MLP checkpoints store activations between linears (``head.0``, ``head.2``, ...).
    Missing or non-linear entries are skipped; the caller checks the resulting count.
    """
    linear_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in source_state_dict.items():
        match = _MLP_HEAD_LINEAR_KEY.match(key)
        if match is None:
            continue
        linear_by_layer.setdefault(int(match.group(1)), {})[match.group(2)] = tensor

    if not linear_by_layer:
        return None

    source_linears: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in sorted(linear_by_layer):
        params = linear_by_layer[layer_idx]
        weight, bias = params.get("weight"), params.get("bias")
        if weight is None or bias is None or weight.dim() != 2 or bias.dim() != 1:
            continue
        source_linears.append((weight, bias))
    return source_linears
