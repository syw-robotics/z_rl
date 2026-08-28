# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy construction helpers for Active Adaptation environments."""

from __future__ import annotations

import copy
import importlib
from typing import Any

_BACKEND_MODULES = {
    "isaac": "active_adaptation.envs.backends.isaac",
    "mujoco": "active_adaptation.envs.backends.mujoco",
    "mjlab": "active_adaptation.envs.backends.mjlab",
    "motrix": "active_adaptation.envs.backends.motrix",
}


def make_active_adaptation_env(
    task_cfg: Any,
    *,
    device: str,
    headless: bool,
    seed: int,
) -> Any:
    """Build an AA ``TransformedEnv`` without constructing an AA policy.

    ``active_adaptation.init(...)`` must be called before this function. Imports
    stay local so importing :mod:`z_rl` does not require AA or TorchRL.
    """
    import active_adaptation as aa

    backend = aa.get_backend()
    if backend is None:
        raise RuntimeError(
            "Active Adaptation is not initialized. Call "
            "active_adaptation.init(cfg, auto_rank=True) before building the environment."
        )
    if backend not in _BACKEND_MODULES:
        raise ValueError(f"Unsupported Active Adaptation backend: {backend!r}")

    importlib.import_module(_BACKEND_MODULES[backend])

    from active_adaptation.envs import _EnvBase
    from torchrl.envs import Compose, InitTracker, StepCounter, TransformedEnv

    task_cfg = copy.deepcopy(task_cfg)
    if backend == "isaac":
        env_class_name = task_cfg.get("env_class", "IsaacBackendEnv")
        env_device = str(device)
    elif backend == "mjlab":
        env_class_name = task_cfg.get("env_class", "MjlabBackendEnv")
        env_device = str(device)
    elif backend == "mujoco":
        env_class_name = task_cfg.get("env_class", "MujocoBackendEnv")
        env_device = "cpu"
    else:
        env_class_name = task_cfg.get("env_class", "MotrixBackendEnv")
        env_device = "cpu"

    try:
        env_cls = _EnvBase.registry[env_class_name]
    except KeyError as exc:
        available = ", ".join(sorted(_EnvBase.registry))
        raise ValueError(
            f"Unknown Active Adaptation environment class {env_class_name!r}. Available classes: {available}"
        ) from exc

    base_env = env_cls(task_cfg, env_device, headless=headless)
    env = TransformedEnv(base_env, Compose(InitTracker(), StepCounter()))
    env.set_seed(seed + aa.get_local_rank())
    return env
