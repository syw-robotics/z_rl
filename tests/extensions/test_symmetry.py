# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for compiled signed-permutation symmetry."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

import pytest

from z_rl.extensions import (
    Symmetry,
    TensorMirrorSpec,
    augment_mirrored_tensor,
    build_obs_group_mirror_spec,
    resolve_symmetry_config,
)
from z_rl.storage import RolloutStorage


def test_tensor_mirror_spec_applies_and_is_involutive() -> None:
    spec = TensorMirrorSpec([1, 0, 2], [-1.0, -1.0, 1.0], name="test")
    value = torch.tensor([[1.0, 2.0, 3.0]])

    mirrored = spec(value)

    assert torch.equal(mirrored, torch.tensor([[-2.0, -1.0, 3.0]]))
    assert torch.equal(spec(mirrored), value)


@pytest.mark.parametrize(
    ("index", "sign", "error"),
    [
        ([0, 0], [1.0, 1.0], "permutation"),
        ([1, 0], [1.0, -1.0], "signs"),
        ([1, 2, 0], [1.0, 1.0, 1.0], "involution"),
    ],
)
def test_tensor_mirror_spec_rejects_invalid_reflections(index, sign, error) -> None:
    with pytest.raises(ValueError, match=error):
        TensorMirrorSpec(index, sign)


def test_build_obs_group_mirror_spec_term_major_history() -> None:
    obs_format = {"policy": {"joint": (2, 2), "clock": (0, 1)}}
    term_specs = {
        "joint": TensorMirrorSpec([1, 0]),
        "clock": TensorMirrorSpec([0], [-1.0]),
    }

    spec = build_obs_group_mirror_spec(
        obs_format,
        "policy",
        term_specs,
        layout_mode="term_major",
    )

    assert spec.index.tolist() == [1, 0, 3, 2, 4]
    assert spec.sign.tolist() == [1.0, 1.0, 1.0, 1.0, -1.0]


def test_build_obs_group_mirror_spec_history_major() -> None:
    obs_format = {"policy": {"joint": (2, 2), "clock": (2, 1)}}
    term_specs = {
        "joint": TensorMirrorSpec([1, 0]),
        "clock": TensorMirrorSpec([0], [-1.0]),
    }

    spec = build_obs_group_mirror_spec(
        obs_format,
        "policy",
        term_specs,
        layout_mode="history_major",
    )

    assert spec.index.tolist() == [1, 0, 2, 4, 3, 5]
    assert spec.sign.tolist() == [1.0, 1.0, -1.0, 1.0, 1.0, -1.0]


def test_build_obs_group_mirror_spec_requires_every_term() -> None:
    obs_format = {"policy": {"known": (0, 1), "missing": (0, 1)}}

    with pytest.raises(ValueError, match="missing symmetry"):
        build_obs_group_mirror_spec(
            obs_format,
            "policy",
            {"known": TensorMirrorSpec([0])},
            layout_mode="term_major",
        )


def test_augment_mirrored_tensor_appends_one_copy() -> None:
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    spec = TensorMirrorSpec([1, 0], [-1.0, -1.0])

    augmented = augment_mirrored_tensor(value, spec)

    assert torch.equal(augmented[:2], value)
    assert torch.equal(augmented[2:], torch.tensor([[-2.0, -1.0], [-4.0, -3.0]]))


def test_symmetry_uses_environment_default_augmentor() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.compiled = False

        def compile_symmetry(self) -> None:
            self.compiled = True

        def augment_symmetry(self, **kwargs):
            return kwargs["obs"], kwargs["actions"]

    env = FakeEnv()

    symmetry = Symmetry(env=env)

    assert env.compiled
    assert symmetry.env is env


def test_resolve_symmetry_augmentation_enables_default_rollout_augmentation() -> None:
    env = object()
    algorithm_cfg = {"symmetry_augmentation": True, "symmetry_cfg": None}

    resolved = resolve_symmetry_config(algorithm_cfg, env)

    assert resolved["symmetry_augmentation"] is True
    assert resolved["symmetry_cfg"] == {"env": env}


def test_resolve_symmetry_augmentation_defaults_to_disabled() -> None:
    resolved = resolve_symmetry_config({"symmetry_cfg": None}, object())

    assert resolved["symmetry_cfg"] is None


def test_symmetry_augments_rollout_minibatch_without_expanding_storage() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.compiled = False

        def compile_symmetry(self) -> None:
            self.compiled = True

        def augment_symmetry(self, env=None, obs=None, actions=None):
            del env
            obs_aug = None
            if obs is not None:
                obs_aug = TensorDict(
                    {key: torch.cat((value, -value), dim=0) for key, value in obs.items()},
                    batch_size=[obs.batch_size[0] * 2],
                )
            actions_aug = None if actions is None else torch.cat((actions, -actions), dim=0)
            return obs_aug, actions_aug

    num_envs = 2
    num_steps = 2
    obs = TensorDict({"policy": torch.arange(4, dtype=torch.float32).reshape(2, 2)}, batch_size=[num_envs])
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [2])
    storage_shape = storage.observations.batch_size
    for step in range(num_steps):
        transition = RolloutStorage.Transition()
        transition.observations = obs.apply(lambda value: value + step * 10.0)
        transition.actions = torch.arange(4, dtype=torch.float32).reshape(num_envs, 2) + step * 10.0
        transition.rewards = torch.zeros(num_envs)
        transition.dones = torch.zeros(num_envs)
        transition.values = torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1) + step * 10.0
        transition.actions_log_prob = torch.arange(num_envs, dtype=torch.float32) + step * 10.0
        transition.distribution_params = (
            torch.zeros(num_envs, 2),
            torch.ones(num_envs, 2),
        )
        storage.add_transition(transition)
    storage.advantages.copy_(torch.arange(4, dtype=torch.float32).reshape(num_steps, num_envs, 1))
    storage.returns.copy_(torch.arange(4, 8, dtype=torch.float32).reshape(num_steps, num_envs, 1))

    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    original_obs = batch.observations.clone()
    original_actions = batch.actions.clone()
    original_values = batch.values.clone()
    original_advantages = batch.advantages.clone()
    original_returns = batch.returns.clone()
    original_log_prob = batch.old_actions_log_prob.clone()
    original_batch_size = num_envs * num_steps
    symmetry = Symmetry(env=FakeEnv(), batch_is_augmented=True)

    symmetry.augment_batch(batch, original_batch_size=original_batch_size)

    assert storage.observations.batch_size == storage_shape
    assert batch.observations.batch_size == torch.Size([original_batch_size * 2])
    assert torch.equal(
        batch.observations["policy"],
        torch.cat((original_obs["policy"], -original_obs["policy"]), dim=0),
    )
    assert torch.equal(batch.actions, torch.cat((original_actions, -original_actions), dim=0))
    assert torch.equal(batch.values, original_values.repeat(2, 1))
    assert torch.equal(batch.advantages, original_advantages.repeat(2, 1))
    assert torch.equal(batch.returns, original_returns.repeat(2, 1))
    assert torch.equal(batch.old_actions_log_prob, original_log_prob.repeat(2, 1))


class _GradTrackingActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.grad_enabled: list[bool] = []

    def forward(self, observations: TensorDict) -> torch.Tensor:
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.linear(observations["policy"])


def _signed_augmentation(
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    augmented_obs = None
    if obs is not None:
        augmented_obs = TensorDict(
            {"policy": torch.cat((obs["policy"], -obs["policy"]), dim=0)},
            batch_size=[obs.batch_size[0] * 2],
        )
    augmented_actions = None if actions is None else torch.cat((actions, -actions), dim=0)
    return augmented_obs, augmented_actions


class _SignedSymmetryEnv:
    def compile_symmetry(self) -> None:
        pass

    def augment_symmetry(
        self,
        obs: TensorDict | None = None,
        actions: torch.Tensor | None = None,
    ) -> tuple[TensorDict | None, torch.Tensor | None]:
        return _signed_augmentation(obs=obs, actions=actions)


def test_detached_mirror_metric_disables_autograd() -> None:
    """A logging-only mirror metric should run its actor forward under no_grad."""
    actor = _GradTrackingActor()
    batch = type("Batch", (), {})()
    batch.observations = TensorDict({"policy": torch.randn(3, 2)}, batch_size=[3])
    symmetry = Symmetry(
        env=_SignedSymmetryEnv(),
        log_mirror_loss=True,
    )

    loss = symmetry.compute_metric(actor, batch, original_batch_size=3)

    assert actor.grad_enabled == [False]
    assert not loss.requires_grad
    assert batch.observations.batch_size == torch.Size([3])


def test_optimized_mirror_loss_keeps_autograd() -> None:
    """An optimized mirror loss should retain its actor gradient graph."""
    actor = _GradTrackingActor()
    batch = type("Batch", (), {})()
    batch.observations = TensorDict({"policy": torch.randn(3, 2)}, batch_size=[3])
    symmetry = Symmetry(
        env=_SignedSymmetryEnv(),
        use_mirror_loss=True,
    )

    loss = symmetry.compute_loss(actor, batch, original_batch_size=3)

    assert actor.grad_enabled == [True]
    assert loss.requires_grad


def test_mirror_metric_logging_config_is_stored() -> None:
    """Mirror diagnostic logging should remain independently configurable."""
    symmetry = Symmetry(
        env=_SignedSymmetryEnv(),
        log_mirror_loss=True,
        mirror_loss_log_interval=3,
    )

    assert symmetry.log_mirror_loss
    assert symmetry.mirror_loss_log_interval == 3
