# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for composite optimizer utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from z_rl.modules import MLP
from z_rl.utils.opt import OptimizerGroup, build_optimizer


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = MLP(4, 2, [8, 8], activation="elu", first_non_muon=True, last_non_muon=True)


def test_mlp_marks_first_and_last_linear_as_non_muon() -> None:
    mlp = MLP(4, 2, [8, 8], activation="elu", first_non_muon=True, last_non_muon=True)
    linear_layers = [layer for layer in mlp if isinstance(layer, nn.Linear)]

    assert getattr(linear_layers[0].weight, "_non_muon", False)
    assert getattr(linear_layers[-1].weight, "_non_muon", False)
    assert not getattr(linear_layers[1].weight, "_non_muon", False)


def test_optimizer_group_steps_all_wrapped_optimizers() -> None:
    model = nn.Linear(2, 2)
    opt_a = torch.optim.SGD(model.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(model.parameters(), lr=0.2)
    group = OptimizerGroup([opt_a, opt_b])

    loss = model(torch.ones(1, 2)).sum()
    group.zero_grad()
    loss.backward()
    group.step()

    assert len(group.state_dict()["optimizers"]) == 2


def test_optimizer_group_updates_learning_rates_after_loading_state() -> None:
    first = nn.Parameter(torch.ones(1))
    second = nn.Parameter(torch.ones(1))
    group = OptimizerGroup([
        torch.optim.SGD([first], lr=0.1),
        torch.optim.SGD([second], lr=0.2),
    ])
    group.load_state_dict(group.state_dict())
    group.param_groups[0]["lr"] = 0.01
    group.param_groups[1]["lr"] = 0.02
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)
    group.step()
    torch.testing.assert_close(first, torch.tensor([0.99]))
    torch.testing.assert_close(second, torch.tensor([0.98]))


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="PyTorch Muon required")
def test_build_optimizer_splits_muon_and_adamw_parameters() -> None:
    model = _TinyModel()
    layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]
    optimizer = build_optimizer([model], learning_rate=1e-3, optimizer="muon_adamw")
    muon, adamw = optimizer.optimizers

    assert isinstance(muon, torch.optim.Muon)
    assert isinstance(adamw, torch.optim.AdamW)
    assert muon.param_groups[0]["params"][0] is layers[1].weight
    assert set(adamw.param_groups[0]["params"]) == set(model.parameters()) - {layers[1].weight}

    state = optimizer.state_dict()
    state["class"] = "MuonAdamWWrapper"
    optimizer.load_state_dict(state)


def test_optimizer_group_rejects_mismatched_checkpoint_before_loading_any_part() -> None:
    first = nn.Parameter(torch.ones(1))
    second = nn.Parameter(torch.ones(1))
    group = OptimizerGroup([
        torch.optim.SGD([first], lr=0.1),
        torch.optim.SGD([second], lr=0.2),
    ])

    with pytest.raises(ValueError, match="optimizers"):
        group.load_state_dict(group.optimizers[0].state_dict())

    checkpoint = group.state_dict()
    with pytest.raises(ValueError, match="checkpoint has 1 optimizers"):
        group.load_state_dict({"optimizers": checkpoint["optimizers"][:1]})

    checkpoint["optimizers"][0]["param_groups"][0]["lr"] = 0.5
    checkpoint["optimizers"][1]["param_groups"][0]["params"] = [0, 1]
    with pytest.raises(ValueError, match="parameter groups do not match"):
        group.load_state_dict(checkpoint)
    assert group.param_groups[0]["lr"] == 0.1


def test_build_optimizer_uses_named_optimizer() -> None:
    model = nn.Linear(2, 2)
    optimizer = build_optimizer([model], learning_rate=1e-3, optimizer="sgd")

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["lr"] == 1e-3


def test_build_optimizer_use_muon_ignores_named_optimizer() -> None:
    if not hasattr(torch.optim, "Muon"):
        return

    model = _TinyModel()
    optimizer = build_optimizer(
        [model],
        learning_rate=1e-3,
        optimizer="sgd",
        use_muon=True,
        muon_adamw_weight_decay=0.05,
    )
    adamw = next(item for item in optimizer.optimizers if isinstance(item, torch.optim.AdamW))

    assert isinstance(optimizer, OptimizerGroup)
    assert not any(isinstance(item, torch.optim.SGD) for item in optimizer.optimizers)
    assert adamw.defaults["weight_decay"] == 0.05
