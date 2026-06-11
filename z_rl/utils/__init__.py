# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions."""

from .log_writer import LogWriter
from .utils import (
    ObsSelector,
    resolve_obs_temporal_selector,
    resolve_target_obs_term_selector,
    resolve_obs_groups,
    inject_obs_time_slice_map,
    get_param,
    check_nan,
    compile_model,
    split_and_pad_trajectories,
    unpad_trajectories,
    resolve_nn_activation,
    resolve_optimizer,
    resolve_callable,
)


def __getattr__(name: str):
    """Lazily import optional logging integrations."""
    if name == "WandbLogWriter":
        from .wandb_utils import WandbLogWriter

        return WandbLogWriter
    if name == "NeptuneLogWriter":
        from .neptune_utils import NeptuneLogWriter

        return NeptuneLogWriter
    raise AttributeError(f"module 'z_rl.utils' has no attribute {name!r}")

__all__ = [
    "LogWriter",
    "NeptuneLogWriter",
    "WandbLogWriter",
    "ObsSelector",
    "resolve_obs_temporal_selector",
    "resolve_target_obs_term_selector",
    "resolve_obs_groups",
    "inject_obs_time_slice_map",
    "get_param",
    "check_nan",
    "compile_model",
    "split_and_pad_trajectories",
    "unpad_trajectories",
    "resolve_nn_activation",
    "resolve_optimizer",
    "resolve_callable",
]
