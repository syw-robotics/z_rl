# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .composition import ComposableModel, GroupObsLatentAdapter, ObsLatentAdapter, HeadSpec, LatentSpec
from .mlp_model import MLPModel
from .variants import (
    CNNModel,
    GroupMLPEncoderModel,
    MLPEncoderModel,
    MoEModel,
    RNNModel,
    SimBaModel,
)
from .variants.cnn_model import CNNLatentSpec
from .variants.group_mlp_encoder_model import GroupMLPLatentSpec
from .variants.mlp_encoder_model import MLPEncoderLatentSpec
from .variants.moe_model import MoEHeadSpec
from .variants.simba_model import SimBaHeadSpec


__all__ = [
    "MLPModel",
    "ComposableModel",
    "ObsLatentAdapter",
    "GroupObsLatentAdapter",
    "LatentSpec",
    "HeadSpec",
    "CNNLatentSpec",
    "CNNModel",
    "MLPEncoderModel",
    "GroupMLPEncoderModel",
    "GroupMLPLatentSpec",
    "MLPEncoderLatentSpec",
    "MoEHeadSpec",
    "MoEModel",
    "RNNModel",
    "SimBaHeadSpec",
    "SimBaModel",
]
