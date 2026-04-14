# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the composable model path."""

from __future__ import annotations

import torch
import torch.nn as nn

from z_rl.models import ComposableModel
from z_rl.models.composition import HeadSpec, LatentSpec
from tests.conftest import make_obs

OBS_GROUPS = {"actor": ["policy"]}


class _ProjectLatent(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class _LatentSpec(LatentSpec):
    def __init__(self, latent_dim: int) -> None:
        self.latent_dim = latent_dim
        self.validated_model = None

    def validate(self, model: nn.Module) -> None:
        self.validated_model = model

    def build_latent_adapter(self, model: nn.Module) -> nn.Module:
        return _ProjectLatent(model.input_dim, self.latent_dim)

    def get_latent_dim(self, model: nn.Module) -> int:
        del model
        return self.latent_dim


class _HeadSpec(HeadSpec):
    def __init__(self) -> None:
        self.validated_model = None
        self.last_input_dim = None
        self.last_output_dim = None

    def validate(self, model: nn.Module) -> None:
        self.validated_model = model

    def build_head(self, model: nn.Module, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        del model, activation
        self.last_input_dim = input_dim
        self.last_output_dim = output_dim
        return nn.Linear(input_dim, output_dim)


class TestComposableModel:
    def test_latent_spec_changes_latent_width(self) -> None:
        obs = make_obs(num_envs=4, obs_dim=8)
        latent_spec = _LatentSpec(latent_dim=5)

        model = ComposableModel(
            obs,
            OBS_GROUPS,
            "actor",
            3,
            hidden_dims=[16],
            latent_spec=latent_spec,
        )

        latent = model.get_latent(obs)

        assert latent_spec.validated_model is model
        assert latent.shape == (4, 5)

    def test_head_spec_receives_updated_latent_dim(self) -> None:
        obs = make_obs(num_envs=4, obs_dim=8)
        latent_spec = _LatentSpec(latent_dim=6)
        head_spec = _HeadSpec()

        model = ComposableModel(
            obs,
            OBS_GROUPS,
            "actor",
            3,
            hidden_dims=[16],
            latent_spec=latent_spec,
            head_spec=head_spec,
        )

        output = model(obs)

        assert head_spec.validated_model is model
        assert head_spec.last_input_dim == 6
        assert head_spec.last_output_dim == 3
        assert output.shape == (4, 3)
