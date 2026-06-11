# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from z_rl.utils import compile_model


def test_compile_model_none_returns_same_module() -> None:
    """Compilation should be opt-in."""
    model = torch.nn.Linear(2, 2)
    assert compile_model(model, None) is model


@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune"])
def test_compile_model_rejects_cuda_graph_modes(mode: str) -> None:
    """CUDA-graph modes are rejected because PPO calls multiple models in sequence."""
    with pytest.raises(ValueError, match="CUDA graphs"):
        compile_model(torch.nn.Linear(2, 2), mode)
