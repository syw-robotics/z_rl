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
        expert_hidden_dims: tuple[int, ...] | list[int],
        gate_hidden_dims: tuple[int, ...] | list[int] | None = None,
        activation: str = "elu",
        top_k: int | None = None,
    ) -> None:
        """Initialize the MoE module.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output dimension for each expert.
            num_experts: Number of experts.
            expert_hidden_dims: Hidden dimensions used by each expert MLP.
            gate_hidden_dims: Hidden dimensions used by the gate MLP. If ``None``, use a single linear layer as gate.
            activation: Activation function used by expert MLPs.
            top_k: Number of experts to evaluate per sample. ``None`` evaluates all experts.
        """
        super().__init__()

        self.num_experts = num_experts
        if top_k is not None and not 1 <= top_k <= num_experts:
            raise ValueError(f"`top_k` must be between 1 and num_experts ({num_experts}), got {top_k}.")
        self.top_k = top_k
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
        # Cached for routing regularizers. Not a buffer: it is not part of the module state.
        self._last_gate_weights = torch.empty(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MoE."""
        # Softmax in fp32 so AMP/bf16 does not sharpen the gate into an accidental one-hot.
        gate_logits = self.gate(x).float()
        if self.top_k is None:
            gate_weights = torch.softmax(gate_logits, dim=-1).to(dtype=x.dtype)
            expert_outputs = self.experts(x)
            output = (gate_weights.unsqueeze(-2) @ expert_outputs).squeeze(-2)
        else:
            top_logits, top_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            top_weights = torch.softmax(top_logits, dim=-1).to(dtype=x.dtype)
            gate_weights = torch.zeros_like(gate_logits, dtype=x.dtype).scatter(-1, top_indices, top_weights)
            expert_outputs = self.experts(x, top_indices)
            output = (top_weights.unsqueeze(-2) @ expert_outputs).squeeze(-2)
        if not torch.onnx.is_in_onnx_export():
            self._last_gate_weights = gate_weights
        if self.output_shape is not None:
            output = output.unflatten(dim=-1, sizes=self.output_shape)
        return output

    def __getstate__(self) -> dict:
        """Drop live router cache so deepcopy/pickle/export cannot copy autograd graphs."""
        state = super().__getstate__()
        cached = state.get("_last_gate_weights")
        if isinstance(cached, torch.Tensor) and cached.grad_fn is not None:
            state["_last_gate_weights"] = torch.empty(0, dtype=cached.dtype, device=cached.device)
        return state

    @property
    def last_gate_weights(self) -> torch.Tensor:
        """Return gate probabilities from the most recent forward pass.

        Returns:
            Tensor of shape ``[..., num_experts]``.
        """
        if self._last_gate_weights.numel() == 0:
            raise RuntimeError("`MoE.forward()` must be called before reading `last_gate_weights`.")
        return self._last_gate_weights

    def gate_entropy(self) -> torch.Tensor:
        """Return the mean entropy of the most recent gate distribution.

        Higher entropy means each sample uses a flatter mixture over experts.
        """
        weights = self._flatten_gate_weights()
        return -torch.special.xlogy(weights, weights).sum(dim=-1).mean()

    def expert_balance_loss(self) -> torch.Tensor:
        """Return a batch-level expert-balance penalty for the most recent gate.

        This is ``KL(mean_gate || uniform)``. It is near zero when experts are used equally
        across the batch, and grows when routing collapses onto a subset of experts.
        """
        mean_w = self._flatten_gate_weights().mean(dim=0)
        return torch.special.xlogy(mean_w, mean_w * self.num_experts).sum()

    def _flatten_gate_weights(self) -> torch.Tensor:
        """Flatten cached gate weights to ``[batch, num_experts]``."""
        weights = self.last_gate_weights
        if weights.dim() == 1:
            return weights.unsqueeze(0)
        return weights.flatten(0, -2)

    def init_distribution_heads(self, distribution: nn.Module) -> None:
        """Initialize expert output heads for distribution-specific parameterization."""
        self.experts.init_distribution_heads(distribution)


class _BatchedMLPExperts(nn.Module):
    """Batched MLP experts implemented as stacked expert parameters for vectorized execution."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
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
            # Keep ``[num_experts, in_dim, out_dim]`` so checkpoints and pretrained-expert loading stay compatible.
            w = nn.Parameter(torch.empty(num_experts, in_dim, out_dim))
            b = nn.Parameter(torch.empty(num_experts, out_dim))
            # Match ``nn.Linear``: kaiming_uniform_(a=sqrt(5)) on a 3D ``[E, I, O]`` tensor would treat it as a
            # convolution and use fan_in = I * O, shrinking weights by sqrt(O).
            bound = in_dim**-0.5
            nn.init.uniform_(w, -bound, bound)
            nn.init.uniform_(b, -bound, bound)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Compute all expert outputs in parallel.

        Args:
            x: Input tensor with shape ``[..., input_dim]``.

        Returns:
            Tensor with shape ``[..., num_experts, output_dim]``.
        """
        if expert_indices is not None:
            return self._forward_selected(x, expert_indices)
        # Keep activations as [E, N, *] so ``torch.bmm`` can use the expert axis as the GEMM batch.
        # ``[N, E, in] @ [E, in, out]`` would treat E as a matrix dim, not a batch dim.
        h = x.reshape(-1, x.shape[-1]).unsqueeze(0).expand(self.num_experts, -1, -1)
        for layer_idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            h = torch.bmm(h, weight) + bias.unsqueeze(1)
            if layer_idx < self.num_layers - 1:
                h = self.activation(h)

        h = h.permute(1, 0, 2).contiguous()
        return h.reshape(*x.shape[:-1], self.num_experts, h.shape[-1])

    def _forward_selected(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Compute only the experts selected for each input sample."""
        flat_x = x.reshape(-1, x.shape[-1])
        flat_indices = expert_indices.reshape(-1, expert_indices.shape[-1])
        outputs = []
        for slot in range(flat_indices.shape[-1]):
            h = flat_x
            selected = flat_indices[:, slot]
            for layer_idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                h = torch.bmm(h.unsqueeze(1), weight.index_select(0, selected)).squeeze(1)
                h = h + bias.index_select(0, selected)
                if layer_idx < self.num_layers - 1:
                    h = self.activation(h)
            outputs.append(h)
        return torch.stack(outputs, dim=1).reshape(
            *x.shape[:-1], expert_indices.shape[-1], self.weights[-1].shape[-1]
        )

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
