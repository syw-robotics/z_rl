# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn

from z_rl.utils import resolve_nn_activation


class MoE(nn.Module):
    """Mixture-of-Experts (MoE) module with MLP experts.

    A linear gating network produces expert weights from the input. Each expert is an MLP that maps from the same
    input space to the same output space. The final output is the weighted sum of expert outputs.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        num_experts: int,
        expert_hidden_dims: tuple[int, ...] | list[int] | int,
        gate_hidden_dims: tuple[int, ...] | list[int] | int | None = None,
        activation: str = "elu",
    ) -> None:
        """Initialize the MoE module.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output dimension for each expert.
            num_experts: Number of experts.
            expert_hidden_dims: Hidden dimensions used by each expert MLP.
            gate_hidden_dims: Hidden dimensions used by the gate MLP. If ``None``, use a single linear layer as gate.
            activation: Activation function used by expert MLPs.
        """
        super().__init__()

        if isinstance(gate_hidden_dims, int):
            gate_hidden_dims = [gate_hidden_dims]

        self.num_experts = num_experts
        if isinstance(output_dim, int):
            self.output_shape: tuple[int, ...] | None = None
            self.output_dim_total = output_dim
        else:
            self.output_shape = tuple(output_dim)
            self.output_dim_total = 1
            for dim in self.output_shape:
                self.output_dim_total *= dim

        if gate_hidden_dims is None:
            self.gate = nn.Linear(input_dim, num_experts)
        else:
            from z_rl.modules.mlp import MLP

            self.gate = MLP(input_dim, num_experts, hidden_dims=gate_hidden_dims, activation=activation)
        self.experts = _BatchedMLPExperts(
            input_dim=input_dim,
            output_dim=self.output_dim_total,
            hidden_dims=expert_hidden_dims,
            num_experts=num_experts,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MoE."""
        # Shape: [..., num_experts]
        gate_weights = torch.softmax(self.gate(x), dim=-1)
        # Shape: [..., num_experts, output_dim_total]
        expert_outputs = self.experts(x)
        gate_weights = gate_weights.unsqueeze(-1)
        # Weighted combination over expert dimension
        output = torch.sum(expert_outputs * gate_weights, dim=-2)
        if self.output_shape is not None:
            output = output.unflatten(dim=-1, sizes=self.output_shape)
        return output

    def init_distribution_heads(self, distribution: nn.Module) -> None:
        """Initialize expert output heads for distribution-specific parameterization."""
        self.experts.init_distribution_heads(distribution)


class _BatchedMLPExperts(nn.Module):
    """Batched MLP experts implemented as stacked expert parameters for vectorized execution."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        num_experts: int,
        activation: str,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.num_experts = num_experts
        self.activation = resolve_nn_activation(activation)
        self.num_layers = len(dims) - 1

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            w = nn.Parameter(torch.empty(num_experts, in_dim, out_dim))
            b = nn.Parameter(torch.empty(num_experts, out_dim))
            nn.init.kaiming_uniform_(w, a=5**0.5)
            fan_in = in_dim
            bound = 1 / fan_in**0.5
            nn.init.uniform_(b, -bound, bound)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all expert outputs in parallel.

        Args:
            x: Input tensor with shape ``[..., input_dim]``.

        Returns:
            Tensor with shape ``[..., num_experts, output_dim]``.
        """
        h = x.reshape(-1, x.shape[-1])

        for layer_idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            if layer_idx == 0:
                h = torch.einsum("bi,eio->beo", h, weight) + bias.unsqueeze(0)
            else:
                h = torch.einsum("bei,eio->beo", h, weight) + bias.unsqueeze(0)

            if layer_idx < self.num_layers - 1:
                h = self.activation(h)

        return h.reshape(x.shape[0], self.num_experts, h.shape[-1])

    @torch.no_grad()
    def init_distribution_heads(self, distribution: nn.Module) -> None:
        """Apply distribution-specific initialization to all expert output heads."""
        # Keep behavior aligned with HeteroscedasticGaussianDistribution.init_head_weights.
        if type(distribution).__name__ != "HeteroscedasticGaussianDistribution":
            return

        output_dim = distribution.output_dim  # type: ignore[attr-defined]
        self.weights[-1][:, :, output_dim:] = 0.0
        if distribution.std_type == "scalar":  # type: ignore[attr-defined]
            self.biases[-1][:, output_dim:] = distribution.init_std  # type: ignore[attr-defined]
        elif distribution.std_type == "log":  # type: ignore[attr-defined]
            init_std_log = torch.log(torch.tensor(distribution.init_std + 1e-7, device=self.biases[-1].device))  # type: ignore[attr-defined]
            self.biases[-1][:, output_dim:] = init_std_log
