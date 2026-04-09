# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .mlp_model import MLPModel
from .moe_model import MoEModel
from .rnn_model import RNNModel

from .mixins import MLPEncoderMixin 


class MLPEncoderMLPModel(MLPEncoderMixin, MLPModel):
    """MLP model composed with :class:`MLPEncoderMixin`."""

    pass


class MLPEncoderMoEModel(MLPEncoderMixin, MoEModel):
    """MoE model composed with :class:`MLPEncoderMixin`."""

    pass


__all__ = [
    "MLPModel",
    "MoEModel",
    "RNNModel",
    "MLPEncoderMLPModel",
    "MLPEncoderMoEModel",
]
