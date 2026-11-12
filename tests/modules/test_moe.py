# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MoE module."""

from __future__ import annotations

import copy

import pytest
import torch

from z_rl.modules import MoE


def _make_moe(num_experts: int = 4, expert_hidden_dims: tuple[int, ...] | list[int] = (16,), **kwargs) -> MoE:
    return MoE(
        input_dim=8,
        output_dim=3,
        num_experts=num_experts,
        expert_hidden_dims=expert_hidden_dims,
        activation="elu",
        **kwargs,
    )


def _einsum_expert_reference(experts: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Reproduce the previous einsum expert path for numerical comparison."""
    h = x.reshape(-1, x.shape[-1])
    for layer_idx, (weight, bias) in enumerate(zip(experts.weights, experts.biases)):
        if layer_idx == 0:
            h = torch.einsum("bi,eio->beo", h, weight) + bias.unsqueeze(0)
        else:
            h = torch.einsum("bei,eio->beo", h, weight) + bias.unsqueeze(0)
        if layer_idx < experts.num_layers - 1:
            h = experts.activation(h)
    return h.reshape(*x.shape[:-1], experts.num_experts, h.shape[-1])


class TestMoERoutingMetrics:
    def test_forward_caches_normalized_gate_weights(self) -> None:
        moe = _make_moe()
        output = moe(torch.randn(5, 8))

        assert output.shape == (5, 3)
        weights = moe.last_gate_weights
        assert weights.shape == (5, 4)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-6)

    def test_reading_gate_weights_before_forward_raises(self) -> None:
        moe = _make_moe()

        with pytest.raises(RuntimeError, match="must be called"):
            _ = moe.last_gate_weights

    def test_uniform_gate_has_near_zero_expert_balance_and_max_entropy(self) -> None:
        moe = _make_moe(num_experts=4)
        moe._last_gate_weights = torch.full((8, 4), 0.25)

        assert moe.expert_balance_loss().item() == pytest.approx(0.0, abs=1e-6)
        assert moe.gate_entropy().item() == pytest.approx(torch.log(torch.tensor(4.0)).item(), abs=1e-6)

    def test_collapsed_gate_has_high_expert_balance_and_near_zero_entropy(self) -> None:
        moe = _make_moe(num_experts=4)
        weights = torch.zeros(8, 4)
        weights[:, 0] = 1.0
        moe._last_gate_weights = weights

        assert moe.expert_balance_loss().item() == pytest.approx(torch.log(torch.tensor(4.0)).item(), abs=1e-5)
        assert moe.gate_entropy().item() == pytest.approx(0.0, abs=1e-5)

    def test_expert_balance_backward_reaches_gate_parameters(self) -> None:
        moe = _make_moe()
        moe(torch.randn(6, 8))
        loss = moe.expert_balance_loss()
        loss.backward()

        assert moe.gate.weight.grad is not None
        assert moe.gate.weight.grad.abs().sum().item() > 0.0


class TestMoEParallelExperts:
    def test_top_k_all_matches_dense_routing(self) -> None:
        dense = _make_moe(num_experts=4)
        sparse = _make_moe(num_experts=4, top_k=4)
        sparse.load_state_dict(dense.state_dict())
        x = torch.randn(5, 8)

        assert torch.allclose(dense(x), sparse(x), atol=1e-6)

    def test_top_k_one_runs_only_selected_expert(self) -> None:
        moe = _make_moe(num_experts=4, top_k=1)
        x = torch.randn(5, 8)
        output = moe(x)
        selected = moe.last_gate_weights.argmax(dim=-1)
        expected = moe.experts(x, selected.unsqueeze(-1)).squeeze(1)

        assert torch.allclose(output, expected, atol=1e-6)
        output.sum().backward()
        for expert_index, weight in enumerate(moe.experts.weights):
            if (selected == expert_index).any():
                assert weight.grad is not None
                assert weight.grad[expert_index].abs().sum() > 0
            else:
                assert weight.grad is not None
                assert weight.grad[expert_index].abs().sum() == 0

    def test_top_k_must_be_in_expert_range(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            _make_moe(num_experts=4, top_k=5)

    def test_mlp_gate_is_accepted(self) -> None:
        moe = _make_moe(gate_hidden_dims=[8])
        assert moe(torch.randn(2, 8)).shape == (2, 3)

    def test_preserves_leading_batch_dims(self) -> None:
        moe = _make_moe()
        output = moe(torch.randn(2, 3, 8))

        assert output.shape == (2, 3, 3)
        assert moe.last_gate_weights.shape == (2, 3, 4)

    def test_matmul_matches_einsum_reference(self) -> None:
        moe = _make_moe()
        x = torch.randn(5, 8)
        assert torch.allclose(moe.experts(x), _einsum_expert_reference(moe.experts, x), atol=1e-6)

    def test_linear_equivalent_init_scale(self) -> None:
        moe = _make_moe()
        first_bound = 8**-0.5
        hidden_bound = 16**-0.5

        assert moe.experts.weights[0].abs().max().item() <= first_bound + 1e-6
        assert moe.experts.biases[0].abs().max().item() <= first_bound + 1e-6
        assert moe.experts.weights[1].abs().max().item() <= hidden_bound + 1e-6
        assert moe.experts.biases[1].abs().max().item() <= hidden_bound + 1e-6

    def test_state_dict_layout_is_unchanged(self) -> None:
        moe = _make_moe()
        state = moe.experts.state_dict()

        assert set(state) == {"weights.0", "weights.1", "biases.0", "biases.1"}
        assert state["weights.0"].shape == (4, 8, 16)
        assert state["weights.1"].shape == (4, 16, 3)
        assert "_last_gate_weights" not in moe.state_dict()

    def test_load_state_dict_restores_outputs(self) -> None:
        source = _make_moe()
        target = _make_moe()
        target.load_state_dict(source.state_dict())
        x = torch.randn(4, 8)
        source.eval()
        target.eval()
        assert torch.allclose(source(x), target(x), atol=1e-6)

    def test_deepcopy_after_forward_does_not_copy_autograd_graph(self) -> None:
        moe = _make_moe()
        moe(torch.randn(4, 8)).sum().backward()
        copied = copy.deepcopy(moe)

        assert copied._last_gate_weights.numel() == 0
        assert copied(torch.randn(2, 8)).shape == (2, 3)
