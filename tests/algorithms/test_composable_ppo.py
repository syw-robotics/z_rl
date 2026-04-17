# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for composable PPO contracts."""

from __future__ import annotations

import torch

from z_rl.algorithms.composition import ComposablePPO, PPOLossSpec
from z_rl.algorithms.ppo import PPO
from z_rl.models import MLPModel
from z_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_actor(obs, obs_groups):
    return MLPModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[32, 32],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def _make_critic(obs, obs_groups):
    return MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[32, 32], activation="elu")


class _DummyLossSpec(PPOLossSpec):
    def __init__(self) -> None:
        self.validated_algo = None

    def validate(self, algo: object) -> None:
        self.validated_algo = algo

    def compute(self, algo: object, minibatch: RolloutStorage.Batch):
        del algo, minibatch
        return {"aux_loss": torch.tensor(2.0)}, {"aux_metric": torch.tensor(3.0)}


class TestComposablePPO:
    def test_requires_loss_spec(self) -> None:
        obs = make_obs(NUM_ENVS, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups)
        critic = _make_critic(obs, obs_groups)
        storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

        try:
            ComposablePPO(actor, critic, storage)
        except ValueError as exc:
            assert "requires a `loss_spec`" in str(exc)
        else:
            raise AssertionError("ComposablePPO should reject construction without a loss_spec.")

    def test_validates_and_merges_loss_spec_outputs(self) -> None:
        obs = make_obs(NUM_ENVS, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups)
        critic = _make_critic(obs, obs_groups)
        storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
        loss_spec = _DummyLossSpec()
        algo = ComposablePPO(actor, critic, storage, loss_spec=loss_spec)

        assert loss_spec.validated_algo is algo

        original_compute_loss = PPO.compute_loss

        def _fake_compute_loss(self, minibatch):
            del self, minibatch
            return {"surrogate_loss": torch.tensor(1.0)}, {"kl": torch.tensor(0.5)}

        PPO.compute_loss = _fake_compute_loss
        try:
            opt_losses, non_opt_losses = algo.compute_loss(minibatch=None)
        finally:
            PPO.compute_loss = original_compute_loss

        assert opt_losses["surrogate_loss"].item() == 1.0
        assert opt_losses["aux_loss"].item() == 2.0
        assert non_opt_losses["kl"].item() == 0.5
        assert non_opt_losses["aux_metric"].item() == 3.0

    def test_subclass_can_override_act(self) -> None:
        class _CustomActionPPO(ComposablePPO):
            def act(self, obs):
                self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
                self.transition.actions = torch.full((NUM_ENVS, NUM_ACTIONS), 9.0)
                self.transition.values = self.critic(obs).detach()
                self.transition.actions_log_prob = torch.zeros(NUM_ENVS)
                self.transition.distribution_params = (torch.ones(NUM_ENVS, NUM_ACTIONS),)
                self.transition.observations = obs
                return self.transition.actions

        obs = make_obs(NUM_ENVS, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups)
        critic = _make_critic(obs, obs_groups)
        storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
        algo = _CustomActionPPO(actor, critic, storage, loss_spec=_DummyLossSpec())

        actions = algo.act(obs)

        assert torch.equal(actions, torch.full((NUM_ENVS, NUM_ACTIONS), 9.0))
