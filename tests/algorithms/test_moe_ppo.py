# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MoEPPO routing regularizers."""

from __future__ import annotations

import pytest
import torch

from tests.conftest import make_obs
from z_rl.algorithms.ppo import PPO
from z_rl.algorithms.variants.moe_ppo import MoEPPO
from z_rl.models import MLPModel, MoEModel
from z_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_moe_actor(obs, obs_groups):
    return MoEModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        num_experts=4,
        expert_hidden_dims=[16],
    )


def _make_mlp_actor(obs, obs_groups):
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


def _make_storage(obs):
    return RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])


def _add_transition(algo: PPO, obs) -> None:
    transition = RolloutStorage.Transition()
    transition.observations = obs
    transition.hidden_states = (None, None)
    transition.actions = algo.actor(obs, stochastic_output=True).detach()
    transition.values = algo.critic(obs).detach()
    transition.actions_log_prob = algo.actor.get_output_log_prob(transition.actions).detach()
    transition.distribution_params = tuple(p.detach() for p in algo.actor.distribution.params)
    transition.rewards = torch.ones(NUM_ENVS)
    transition.dones = torch.zeros(NUM_ENVS)
    algo.storage.add_transition(transition)


class TestMoEPPO:
    def test_rejects_models_without_moe_heads(self) -> None:
        obs = make_obs(NUM_ENVS, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_mlp_actor(obs, obs_groups)
        critic = _make_critic(obs, obs_groups)
        storage = _make_storage(obs)

        with pytest.raises(ValueError, match="requires the actor or critic head to be `MoE`"):
            MoEPPO(actor, critic, storage)

    def test_compute_loss_includes_routing_terms(self) -> None:
        obs = make_obs(NUM_ENVS, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_moe_actor(obs, obs_groups)
        critic = _make_critic(obs, obs_groups)
        storage = _make_storage(obs)
        algo = MoEPPO(actor, critic, storage, schedule="fixed")

        assert algo.gate_entropy_loss_coef == 0.0
        assert algo.expert_balance_loss_coef == 1.0e-4

        for _ in range(NUM_STEPS):
            _add_transition(algo, obs)
        algo.compute_returns(obs)
        minibatch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
        opt_losses, non_opt_losses = algo.compute_loss(minibatch)

        assert "gate_entropy_loss" in opt_losses
        assert "expert_balance_loss" in opt_losses
        assert "moe_gate_entropy" in non_opt_losses
        assert torch.isfinite(opt_losses["expert_balance_loss"])
        assert torch.isfinite(non_opt_losses["moe_gate_entropy"])
        assert torch.allclose(opt_losses["gate_entropy_loss"], -non_opt_losses["moe_gate_entropy"])
