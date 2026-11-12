# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from z_rl.models.composition import ComposableModel
from z_rl.models.mlp_model import ObservationNormalizationConfig, _as_export_latent_adapter
from z_rl.modules import RNN, HiddenState


class RNNModel(ComposableModel):
    """Recurrent ``ComposableModel`` backbone.

    Data flow: ``obs TensorDict -> latent adapter -> RNN -> head -> (distribution) -> output``.

    The latent adapter produces the RNN input from structured observations. The default adapter concatenates
    1D observation groups. Pass ``latent_spec`` (for example ``CNNLatentSpec``) or ``head_spec`` to compose
    vision encoding or a custom head without turning the RNN itself into a spec.
    """

    is_recurrent: bool = True
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: ObservationNormalizationConfig = False,
        distribution_cfg: dict | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        latent_spec: Any = None,
        head_spec: Any = None,
        **kwargs,
    ) -> None:
        """Initialize the RNN-based model."""
        self.rnn_type = rnn_type
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            latent_spec=latent_spec,
            head_spec=head_spec,
            **kwargs,
        )

    def build_latent_adapter(self) -> nn.Module:
        """Build the latent adapter, then size the RNN from the adapter output.

        The adapter (or ``latent_spec``) determines the RNN input width. The head then consumes
        ``rnn_hidden_dim``, not the adapter width.
        """
        adapter = super().build_latent_adapter()
        # define the rnn module
        self.rnn = RNN(int(self.latent_dim), self.rnn_hidden_dim, self.rnn_num_layers, self.rnn_type)
        # head input dim is the rnn hidden dim
        self.latent_dim = self.rnn_hidden_dim
        return adapter

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by passing adapter output through the RNN."""
        latent = super().get_latent(obs)
        # append a rnn module after the latent adapter, before the head
        latent = self.rnn(latent, masks, hidden_state).squeeze(0)
        return latent

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the recurrent hidden state of the RNN."""
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state of the RNN."""
        return self.rnn.hidden_state  # type: ignore

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state for truncated backpropagation."""
        self.rnn.detach_hidden_state(dones)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export.

        Export is vector-obs only: one concatenated ``obs`` tensor plus recurrent state
        (``h_in`` / ``c_in``). Extra adapter inputs such as a CNN image are not packed here.
        """
        return _OnnxRNNModel(self, verbose)


class _OnnxRNNModel(nn.Module):
    """Exportable RNN model for ONNX."""

    is_recurrent: bool = True

    def __init__(self, model: RNNModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an RNNModel."""
        super().__init__()
        self.verbose = verbose
        self.latent_adapter = _as_export_latent_adapter(model.latent_adapter)
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.head = copy.deepcopy(model.head)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        if isinstance(self.rnn, nn.LSTM):
            self.rnn_type = "lstm"
        elif isinstance(self.rnn, nn.GRU):
            self.rnn_type = "gru"
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn)}")

        self.input_size = model.obs_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run deterministic inference for ONNX export."""
        latent = self.latent_adapter(obs)
        if self.rnn_type == "lstm":
            latent, (h_out, c_out) = self.rnn(latent.unsqueeze(0), (h_in, c_in))
            actions = self.deterministic_output(self.head(latent.squeeze(0)))
            return actions, h_out, c_out
        latent, h_out = self.rnn(latent.unsqueeze(0), h_in)
        actions = self.deterministic_output(self.head(latent.squeeze(0)))
        return actions, h_out, None

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        obs = torch.zeros(1, self.input_size)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
            return (obs, h_in, c_in)
        return (obs, h_in)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        if self.rnn_type == "lstm":
            return ["obs", "h_in", "c_in"]
        return ["obs", "h_in"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        if self.rnn_type == "lstm":
            return ["actions", "h_out", "c_out"]
        return ["actions", "h_out"]
