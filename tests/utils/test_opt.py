# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for composite optimizer utilities."""

from __future__ import annotations

import torch
import torch.nn as nn

from z_rl.modules import MLP
from z_rl.utils.opt import MuonAdamWWrapper, OptimizerGroup, build_optimizer


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


def test_muon_adamw_wrapper_splits_parameters() -> None:
    if not hasattr(torch.optim, "Muon"):
        return

    model = _TinyModel()
    wrapper = MuonAdamWWrapper([model], lr=1e-3)
    state = wrapper.state_dict()

    assert len(state["optimizers"]) == 2


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

    assert isinstance(optimizer, MuonAdamWWrapper)
    assert not any(isinstance(item, torch.optim.SGD) for item in optimizer.optimizers)
    assert adamw.defaults["weight_decay"] == 0.05
