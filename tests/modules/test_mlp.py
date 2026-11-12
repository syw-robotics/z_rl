# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MLP modules."""

from __future__ import annotations

from typing import Literal

import pytest
import torch
import torch.nn as nn

from z_rl.modules import MLP


class TestMLP:
    """Tests for ``MLP``."""

    def test_layer_norm_is_disabled_by_default(self) -> None:
        """The default architecture should remain unchanged for checkpoint compatibility."""
        mlp = MLP(input_dim=4, output_dim=2, hidden_dims=[8, 6], activation="relu")

        assert [type(layer) for layer in mlp] == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]

    @pytest.mark.parametrize(
        ("layer_norm", "expected_types"),
        [
            (
                "pre_activation",
                [nn.Linear, nn.LayerNorm, nn.ReLU, nn.Linear, nn.LayerNorm, nn.ReLU, nn.Linear],
            ),
            (
                "post_activation",
                [nn.Linear, nn.ReLU, nn.LayerNorm, nn.Linear, nn.ReLU, nn.LayerNorm, nn.Linear],
            ),
        ],
    )
    def test_layer_norm_position(
        self,
        layer_norm: Literal["pre_activation", "post_activation"],
        expected_types: list[type[nn.Module]],
    ) -> None:
        """LayerNorm should be positioned around hidden activations and omitted from the output layer."""
        mlp = MLP(input_dim=4, output_dim=2, hidden_dims=[8, 6], activation="relu", layer_norm=layer_norm)

        assert [type(layer) for layer in mlp] == expected_types
        assert mlp(torch.randn(3, 4)).shape == (3, 2)

    def test_init_weights_counts_only_linear_layers(self) -> None:
        """Per-layer gains should remain aligned with Linear layers when LayerNorm is enabled."""
        mlp = MLP(input_dim=4, output_dim=2, hidden_dims=[8, 6], layer_norm="pre_activation")

        mlp.init_weights((1.0, 0.5, 0.1))

        assert all(torch.count_nonzero(layer.bias) == 0 for layer in mlp if isinstance(layer, nn.Linear))

    def test_invalid_layer_norm_position_raises(self) -> None:
        """Invalid LayerNorm positions should fail during construction."""
        with pytest.raises(ValueError, match="Unsupported LayerNorm position"):
            MLP(input_dim=4, output_dim=2, hidden_dims=[8], layer_norm="invalid")  # type: ignore[arg-type]
