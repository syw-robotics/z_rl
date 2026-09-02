# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from z_rl.adaptor.active_adaptation import env_factory, make_active_adaptation_env


def _install_factory_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str | None,
) -> tuple[type, list[str]]:
    aa_module = ModuleType("active_adaptation")
    aa_module.get_backend = lambda: backend
    aa_module.get_local_rank = lambda: 2
    monkeypatch.setitem(sys.modules, "active_adaptation", aa_module)

    class FakeBackendEnv:
        def __init__(self, cfg: Any, device: str, *, headless: bool) -> None:
            self.cfg = cfg
            self.device = device
            self.headless = headless

    registry = {
        "IsaacBackendEnv": FakeBackendEnv,
        "MjlabBackendEnv": FakeBackendEnv,
        "MujocoBackendEnv": FakeBackendEnv,
        "MotrixBackendEnv": FakeBackendEnv,
    }
    aa_envs_module = ModuleType("active_adaptation.envs")
    aa_envs_module._EnvBase = SimpleNamespace(registry=registry)
    monkeypatch.setitem(sys.modules, "active_adaptation.envs", aa_envs_module)

    class FakeTransformedEnv:
        def __init__(self, base_env: Any, transform: Any) -> None:
            self.base_env = base_env
            self.transform = transform
            self.seed = None

        def set_seed(self, seed: int) -> None:
            self.seed = seed

    torchrl_envs_module = ModuleType("torchrl.envs")
    torchrl_envs_module.Compose = lambda *transforms: transforms
    torchrl_envs_module.InitTracker = type("InitTracker", (), {})
    torchrl_envs_module.StepCounter = type("StepCounter", (), {})
    torchrl_envs_module.TransformedEnv = FakeTransformedEnv
    torchrl_module = ModuleType("torchrl")
    torchrl_module.envs = torchrl_envs_module
    monkeypatch.setitem(sys.modules, "torchrl", torchrl_module)
    monkeypatch.setitem(sys.modules, "torchrl.envs", torchrl_envs_module)

    imported_modules = []
    monkeypatch.setattr(
        env_factory.importlib,
        "import_module",
        lambda name: imported_modules.append(name),
    )
    return FakeTransformedEnv, imported_modules


def test_factory_requires_aa_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject environment construction before AA selects a backend."""
    _install_factory_modules(monkeypatch, backend=None)

    with pytest.raises(RuntimeError, match="Active Adaptation is not initialized"):
        make_active_adaptation_env({}, device="cuda:0", headless=True, seed=42)


def test_factory_builds_registered_backend_without_aa_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct the registered backend, TorchRL transforms, and rank-local seed."""
    transformed_env_cls, imported_modules = _install_factory_modules(monkeypatch, backend="isaaclab")
    task_cfg = {"name": "fake-task"}

    env = make_active_adaptation_env(
        task_cfg,
        device="cuda:2",
        headless=True,
        seed=40,
    )

    assert isinstance(env, transformed_env_cls)
    assert env.base_env.cfg == task_cfg
    assert env.base_env.cfg is not task_cfg
    assert env.base_env.device == "cuda:2"
    assert env.base_env.headless is True
    assert env.seed == 42
    assert imported_modules == ["active_adaptation.envs.backends.isaaclab"]
