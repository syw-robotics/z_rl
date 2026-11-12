# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from z_rl.modules import MLP, EmpiricalNormalization, HiddenState
from z_rl.modules.distribution import Distribution
from z_rl.models.composition.adapters import ObsLatentAdapter
from z_rl.utils import ObsSelector, resolve_class, unpad_trajectories

# IsaacLab normalization config classes become dictionaries before reaching core models; booleans preserve the
# concise disabled/default-normalizer interface.
ObservationNormalizationConfig = bool | dict[str, object]


class MLPModel(nn.Module):
    """MLP-based neural model.

    Data flow: ``obs TensorDict -> latent adapter -> head -> (distribution) -> output``.

    The default latent adapter preserves the historical behavior by concatenating active 1D observation groups and
    optionally normalizing them. Custom latent adapters may instead consume the structured TensorDict directly before
    returning the latent tensor consumed by the head. Image groups are allowed only when a `latent_spec` consumes them.
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
        layer_norm: Literal["pre_activation", "post_activation"] | None = None,
        obs_normalization: ObservationNormalizationConfig = False,
        distribution_cfg: dict | None = None,
        obs_group_time_slice_map: dict[str, dict[str, ObsSelector]] | None = None,
        obs_format: dict[str, dict[str, tuple[int, ...]]] | None = None,
    ) -> None:
        """Initialize the MLP-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            layer_norm: LayerNorm position in each hidden layer of the default MLP head. None disables it.
            obs_normalization: False disables normalization, True uses its defaults, and a dictionary configures
                ``EmpiricalNormalization``.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
            obs_group_time_slice_map: Cached time-slice metadata, typically from ``VecEnv.obs_group_time_slice_map``.
            obs_format: Cached per-group term shapes, typically from ``VecEnv.obs_format``.
        """
        super().__init__()

        # Observation related attributes
        self.obs_groups = obs_groups[obs_set]
        self.obs_group_shapes = {group: tuple(obs[group].shape[1:]) for group in self.obs_groups}
        allow_image_groups = getattr(self, "latent_spec", None) is not None
        self.obs_dim = 0
        vector_dims: list[int] = []
        for obs_group in self.obs_groups:
            ndim = len(obs[obs_group].shape)
            if ndim == 2:
                group_dim = int(obs[obs_group].shape[-1])
                self.obs_dim += group_dim
                vector_dims.append(group_dim)
            elif ndim == 4 and allow_image_groups:
                continue
            else:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
        self.obs_group_dims = tuple(vector_dims)
        self.input_dim = self.obs_dim
        self.obs_group_time_slice_map = obs_group_time_slice_map or {}
        self.obs_format = obs_format or {}
        self.obs_normalization = obs_normalization
        self.layer_norm = layer_norm
        if not hasattr(self, "latent_dim"):
            self.latent_dim = self.obs_dim

        # Distribution initialization
        if distribution_cfg is None:
            self.distribution = None
            head_output_dim = output_dim
        else:
            dist_class, dist_cfg = resolve_class(distribution_cfg)
            self.distribution: Distribution = dist_class(output_dim, **dist_cfg)  # type: ignore[assignment]
            head_output_dim = self.distribution.input_dim

        # Build the latent adapter and head
        self.latent_adapter = self.build_latent_adapter()
        self._update_normalization = getattr(self.latent_adapter, "update_normalization", None)
        self.head = self.build_head(self.latent_dim, head_output_dim, hidden_dims, activation)
        if self.distribution is not None:  # init distribution-specific weights in the Head module
            self.distribution.init_head_weights(self.head)

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
        output, _ = self.forward_with_context(obs, masks, hidden_state, stochastic_output)
        return output

    def forward_with_context(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Run a forward pass and return the model output with optional intermediates.

        Returns:
            output: Model output. A sample when a distribution is configured and ``stochastic_output`` is true;
                the deterministic distribution output when a distribution is configured and ``stochastic_output`` is
                false; the raw head output when no distribution is configured.
            context: Intermediate tensors reused by update-time losses, including the latent passed to the head.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Get MLP input latent
        latent = self.get_latent(obs, masks, hidden_state)
        # MLP forward pass
        mlp_output = self.head(latent)
        context = {"latent": latent}
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample(), context
            return self.distribution.deterministic_output(mlp_output), context
        return mlp_output, context

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
        if self._update_normalization is not None:
            self._update_normalization(obs)

    def module_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return one state dict per direct child.

        Children are ``latent_adapter``, ``head``, and, when present, ``rnn`` and ``distribution``.
        Names inside each dict are relative to that child.
        """
        return {name: child.state_dict() for name, child in self.named_children()}

    def load_module_state_dict(
        self, module_name: str, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        """Load one direct child from a module state dict."""
        children = dict(self.named_children())
        if module_name not in children:
            raise KeyError(f"Module `{module_name}` is not a direct child. Available: {sorted(children)}.")
        children[module_name].load_state_dict(state_dict, strict=strict)

    def build_latent_adapter(self) -> nn.Module:
        """Build the latent adapter that maps observations to the head input.
        This barely concatenates and normalizes observation groups.
        """
        if self.obs_normalization is False:
            obs_normalizer: nn.Module = nn.Identity()
        else:
            normalization_cfg = {} if self.obs_normalization is True else self.obs_normalization
            obs_normalizer = EmpiricalNormalization(self.obs_dim, **normalization_cfg)
        return ObsLatentAdapter(
            obs_groups=self.obs_groups,
            obs_normalizer=obs_normalizer,
        )

    def build_head(
        self, input_dim: int, output_dim: int | list[int], hidden_dims: tuple[int, ...] | list[int], activation: str
    ) -> nn.Module:
        """Build the output head that consumes the model latent.
        This is a simple MLP network.
        """
        # When use_muon=True, hidden layers use Muon; input/output linear layers stay on AdamW.
        return MLP(
            input_dim,
            output_dim,
            hidden_dims,
            activation,
            layer_norm=self.layer_norm,
            first_non_muon=True,
            last_non_muon=True,
        )


"""
Export Utils
"""


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
        export_dummy_inputs = getattr(model.latent_adapter, "export_dummy_inputs", None)
        export_input_names = getattr(model.latent_adapter, "export_input_names", None)
        self._dummy_inputs = (
            export_dummy_inputs() if export_dummy_inputs is not None else (torch.zeros(1, self.input_size),)
        )
        self._input_names = list(export_input_names()) if export_input_names is not None else ["obs"]

    def forward(self, *obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent = self.latent_adapter(*obs)
        out = self.head(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        return self._dummy_inputs

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
