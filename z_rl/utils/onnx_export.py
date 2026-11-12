# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect

import torch

_EXPORT_SUPPORTS_EXTERNAL_DATA = "external_data" in inspect.signature(torch.onnx.export).parameters


def export_to_onnx(*args, **kwargs) -> None:
    """Export a module to ONNX, keeping weights in a single file when supported."""
    if "external_data" not in kwargs and _EXPORT_SUPPORTS_EXTERNAL_DATA:
        kwargs["external_data"] = False
    torch.onnx.export(*args, **kwargs)
