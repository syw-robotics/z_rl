# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from z_rl.modules import MLP, EmpiricalNormalization, HiddenState
from z_rl.modules.distribution import Distribution
from z_rl.utils import ObsSelector, resolve_callable, unpad_trajectories


class MLPModel(nn.Module):
    """MLP-based neural model.

    Data flow: ``obs TensorDict -> latent adapter -> head -> (distribution) -> output``.

    The default latent adapter preserves the historical behavior by concatenating active 1D observation groups and
    optionally normalizing them. Custom latent adapters may instead consume the structured TensorDict directly before
    returning the latent tensor consumed by the head.
    """

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        obs_group_time_slice_map: dict[str, dict[str, ObsSelector]] | None = None,
    ) -> None:
        """Initialize the MLP-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
            obs_group_time_slice_map: Cached time-slice metadata, typically from ``VecEnv.obs_group_time_slice_map``.
        """
        super().__init__()

        self._init_observation_pipeline(obs, obs_groups, obs_set, obs_normalization, obs_group_time_slice_map)
        self.distribution, head_output_dim = self._build_distribution(output_dim, distribution_cfg)
        self.latent_adapter = self.build_latent_adapter()
        self.head = self.build_head(self.get_latent_dim(), head_output_dim, hidden_dims, activation)
        self.init_head_weights()

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MLP model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Get MLP input latent
        latent = self.get_latent(obs, masks, hidden_state)
        # MLP forward pass
        mlp_output = self.head(latent)
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent from the structured observation TensorDict."""
        return self.latent_adapter(obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the internal state for recurrent models (no-op)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None`` for MLP)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach therecurrent hidden state for truncated backpropagation (no-op)."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of outputs under the current distribution."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two parameterizations of the distribution."""
        return self.distribution.kl_divergence(old_params, new_params)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMLPModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics from a batch of observations."""
        update = getattr(self.latent_adapter, "update_normalization", None)
        if update is not None:
            update(obs)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the model head."""
        return self.obs_dim

    def _init_observation_pipeline(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        obs_normalization: bool,
        obs_group_time_slice_map: dict[str, dict[str, ObsSelector]] | None,
    ) -> None:
        """Resolve observation metadata and build the normalization stage."""
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.input_dim = self.obs_dim
        self.obs_group_time_slice_map = obs_group_time_slice_map or {}
        self.obs_normalization = obs_normalization

    def _build_obs_normalizer(self, obs_normalization: bool) -> nn.Module:
        """Build the observation normalizer used before latent construction."""
        if obs_normalization:
            return EmpiricalNormalization(self.obs_dim)
        return torch.nn.Identity()

    def _build_distribution(
        self, output_dim: int, distribution_cfg: dict | None
    ) -> tuple[Distribution | None, int | list[int]]:
        """Build the optional output distribution and return its required head output dimension."""
        if distribution_cfg is None:
            return None, output_dim

        dist_cfg = dict(distribution_cfg)
        dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore
        distribution = dist_class(output_dim, **dist_cfg)
        return distribution, distribution.input_dim

    def build_latent_adapter(self) -> nn.Module:
        """Build the latent adapter that maps observations to the head input."""
        return _FlatNormalizedLatentAdapter(
            obs_groups=self.obs_groups,
            obs_dim=self.obs_dim,
            obs_normalizer=self._build_obs_normalizer(self.obs_normalization),
        )

    def build_head(
        self, input_dim: int, output_dim: int | list[int], hidden_dims: tuple[int, ...] | list[int], activation: str
    ) -> nn.Module:
        """Build the output head that consumes the model latent."""
        return MLP(input_dim, output_dim, hidden_dims, activation)

    def init_head_weights(self) -> None:
        """Initialize distribution-specific head weights after head construction."""
        if self.distribution is not None:
            self.distribution.init_head_weights(self.head)

    @property
    def obs_normalizer(self) -> nn.Module:
        """Return the normalizer owned by the latent adapter, when present."""
        normalizer = getattr(self.latent_adapter, "obs_normalizer", None)
        if normalizer is None:
            raise AttributeError(f"{type(self.latent_adapter).__name__} does not expose 'obs_normalizer'.")
        return normalizer


"""
Export Utils
"""


class _FlatNormalizedLatentAdapter(nn.Module):
    """Default latent adapter: active obs groups -> flat tensor -> optional normalization."""

    def __init__(self, obs_groups: list[str], obs_dim: int, obs_normalizer: nn.Module) -> None:
        super().__init__()
        self.obs_groups = obs_groups
        self.obs_dim = obs_dim
        self.obs_normalizer = obs_normalizer

    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate configured observation groups and normalize the flat tensor."""
        return self.obs_normalizer(self._flatten_obs(obs))

    def update_normalization(self, obs: TensorDict) -> None:
        """Update running normalization statistics from structured observations."""
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.update(self._flatten_obs(obs))

    def as_export_module(self) -> nn.Module:
        """Return a tensor-only latent adapter for ONNX export."""
        return _FlatNormalizedLatentAdapterExporter(copy.deepcopy(self.obs_normalizer))

    def _flatten_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)


class _FlatNormalizedLatentAdapterExporter(nn.Module):
    """Tensor-only export module for the default latent adapter."""

    def __init__(self, obs_normalizer: nn.Module) -> None:
        super().__init__()
        self.obs_normalizer = obs_normalizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a pre-concatenated observation tensor."""
        return self.obs_normalizer(x)


def _as_export_latent_adapter(latent_adapter: nn.Module) -> nn.Module:
    """Return a tensor-only copy of a runtime latent adapter for ONNX export."""
    export_adapter = getattr(latent_adapter, "as_export_module", None)
    if export_adapter is not None:
        return copy.deepcopy(export_adapter())
    # Fallback is intentionally narrow: custom adapters without as_export_module()
    # must already accept the flat tensor passed by the ONNX wrapper.
    return copy.deepcopy(latent_adapter)


class _OnnxMLPModel(nn.Module):
    """Exportable MLP model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: MLPModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an MLPModel."""
        super().__init__()
        self.verbose = verbose
        self.latent_adapter = _as_export_latent_adapter(model.latent_adapter)
        self.head = copy.deepcopy(model.head)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent = self.latent_adapter(x)
        out = self.head(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
