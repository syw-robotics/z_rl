# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapters for using Active Adaptation environments with Z-RL.

NOTE: no real AA/sim tests yet. ``tests/adaptor/`` is fake-env only.
"""

from .env_factory import make_active_adaptation_env
from .symmetry import ActiveAdaptationSymmetryCompiler
from .vecenv_wrapper import ActiveAdaptationVecEnvWrapper

__all__ = [
    "ActiveAdaptationSymmetryCompiler",
    "ActiveAdaptationVecEnvWrapper",
    "make_active_adaptation_env",
]
