# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from functools import reduce
from typing import Literal

import torch
import torch.nn as nn

from z_rl.utils import get_param, resolve_nn_activation


class MLP(nn.Sequential):
    """Multi-Layer Perceptron.

    The MLP network is a sequence of linear layers and activation functions. The last layer is a linear layer that
    outputs the desired dimension unless the last activation function is specified.

    It provides additional conveniences:
    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
        layer_norm: Literal["pre_activation", "post_activation"] | None = None,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            activation: Activation function.
            last_activation: Activation function of the last layer. None results in a linear last layer.
            layer_norm: Position of LayerNorm relative to the activation function in each hidden layer. None disables
                LayerNorm.

        Raises:
            ValueError: If ``layer_norm`` is not None, ``"pre_activation"``, or ``"post_activation"``.
        """
        super().__init__()

        if layer_norm not in (None, "pre_activation", "post_activation"):
            raise ValueError(
                f"Unsupported LayerNorm position: {layer_norm}. Supported values are None, 'pre_activation', and "
                "'post_activation'."
            )

        # Resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        if layer_norm == "pre_activation":
            layers.append(nn.LayerNorm(hidden_dims_processed[0]))
        layers.append(activation_mod)
        if layer_norm == "post_activation":
            layers.append(nn.LayerNorm(hidden_dims_processed[0]))

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            if layer_norm == "pre_activation":
                layers.append(nn.LayerNorm(hidden_dims_processed[layer_index + 1]))
            layers.append(activation_mod)
            if layer_norm == "post_activation":
                layers.append(nn.LayerNorm(hidden_dims_processed[layer_index + 1]))

        # Add last layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            # Compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # Add a layer to reshape the output to the desired shape
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Add last activation function if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, scales: float | tuple[float, ...]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """
        linear_idx = 0
        for module in self:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, linear_idx))
                nn.init.zeros_(module.bias)
                linear_idx += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        for layer in self:
            x = layer(x)
        return x
