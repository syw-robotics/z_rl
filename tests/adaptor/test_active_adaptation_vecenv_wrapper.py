# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warnings
from pathlib import Path
from tensordict import TensorDict
from typing import Any, Literal, NoReturn

import pytest

from z_rl.adaptor.active_adaptation import ActiveAdaptationVecEnvWrapper
from z_rl.runners import OnPolicyRunner


class _ActionManager:
    action_dim = 2


class _BaseEnv:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.cfg = {"name": "fake-aa"}
        self.input_managers = {"action": _ActionManager()}
        self.max_episode_length = torch.full((num_envs, 1), 8, dtype=torch.long)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)


class _FakeAAEnv:
    def __init__(self, num_envs: int = 4, discount: float = 1.0) -> None:
        self.base_env = _BaseEnv(num_envs)
        self.device = self.base_env.device
        self.discount = discount
        self.reset_calls = 0
        self.step_calls = 0
        self.last_actions = None
        self.closed = False

    def reset(self) -> TensorDict:
        self.reset_calls += 1
        return TensorDict(
            {
                "policy": torch.zeros(self.base_env.num_envs, 4),
                "priv": torch.ones(self.base_env.num_envs, 2),
                "episode_id": torch.arange(self.base_env.num_envs),
            },
            batch_size=[self.base_env.num_envs],
        )

    def step_and_maybe_reset(self, carry: TensorDict) -> tuple[TensorDict, TensorDict]:
        num_envs = self.base_env.num_envs
        self.step_calls += 1
        self.last_actions = carry["action"].clone()
        self.base_env.episode_length_buf += 1

        truncated = torch.zeros(num_envs, 1, dtype=torch.bool)
        terminated = torch.zeros(num_envs, 1, dtype=torch.bool)
        truncated[1] = True
        terminated[2] = True
        done = truncated | terminated

        policy = carry["policy"] + 1.0
        priv = carry["priv"] + 2.0
        reward = TensorDict(
            {
                "loco": torch.arange(1, num_envs + 1, dtype=torch.float32).unsqueeze(
                    -1
                ),
                "aux": torch.full((num_envs, 1), 0.5),
            },
            batch_size=[num_envs],
        )
        stats = TensorDict(
            {
                "loco": TensorDict(
                    {
                        "track_velocity": torch.arange(
                            1, 1 + num_envs, dtype=torch.float32
                        ).unsqueeze(-1),
                        "return": torch.arange(
                            10, 10 + num_envs, dtype=torch.float32
                        ).unsqueeze(-1),
                    },
                    batch_size=[num_envs],
                ),
                "termination": TensorDict(
                    {"time_out": truncated.float(), "fallen": terminated.float()},
                    batch_size=[num_envs],
                ),
                "episode_len": self.base_env.episode_length_buf.float().unsqueeze(-1),
            },
            batch_size=[num_envs],
        )
        next_td = TensorDict(
            {
                "policy": policy,
                "priv": priv,
                "reward": reward,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
                "discount": torch.full((num_envs, 1), self.discount),
                "stats": stats,
            },
            batch_size=[num_envs],
        )
        transition = TensorDict({"next": next_td}, batch_size=[num_envs])

        next_carry = TensorDict(
            {"policy": policy.clone(), "priv": priv.clone()},
            batch_size=[num_envs],
        )
        next_carry[done.squeeze(-1)] = 0.0
        self.base_env.episode_length_buf[done.squeeze(-1)] = 0
        return transition, next_carry

    def set_seed(self, seed: int) -> int:
        return seed

    def close(self) -> None:
        self.closed = True


def _make_wrapper(
    env: _FakeAAEnv | None = None,
    *,
    discount_mode: Literal["ignore", "warn", "error"] = "ignore",
    clip_actions: float | None = None,
) -> ActiveAdaptationVecEnvWrapper:
    return ActiveAdaptationVecEnvWrapper(
        env or _FakeAAEnv(),
        observation_keys=("policy", "priv"),
        reward_keys=("loco", "aux"),
        clip_actions=clip_actions,
        discount_mode=discount_mode,
        include_episode_stats=True,
    )


def _train_config() -> dict[str, Any]:
    return {
        "num_steps_per_env": 2,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "activation": "elu",
        },
    }


def test_wrapper_maps_aa_transition_to_z_rl_contract() -> None:
    """Map AA carry and transition fields to the Z-RL vector-env contract."""
    aa_env = _FakeAAEnv()
    env = _make_wrapper(aa_env, clip_actions=0.25)

    assert aa_env.reset_calls == 1
    assert set(env.get_observations().keys()) == {"policy", "priv"}
    assert env.num_actions == 2
    assert env.max_episode_length == 8

    observations, rewards, dones, extras = env.step(torch.ones(4, 2))

    torch.testing.assert_close(aa_env.last_actions, torch.full((4, 2), 0.25))
    torch.testing.assert_close(rewards, torch.tensor([1.5, 2.5, 3.5, 4.5]))
    torch.testing.assert_close(dones, torch.tensor([0, 1, 1, 0]))
    torch.testing.assert_close(
        extras["time_outs"], torch.tensor([False, True, False, False])
    )
    torch.testing.assert_close(observations["policy"][0], torch.ones(4))
    torch.testing.assert_close(observations["policy"][1], torch.zeros(4))
    assert set(extras["log"]) == {
        "Episode_Reward/loco/track_velocity",
        "Episode_Reward/loco/return",
        "Episode_Termination/time_out",
        "Episode_Termination/fallen",
        "Episode/episode_len",
    }
    assert extras["log_on_done"] is True
    assert extras["log_scale"]["Episode_Reward/loco/track_velocity"] == 1 / 8
    torch.testing.assert_close(
        extras["log"]["Episode_Reward/loco/track_velocity"],
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )
    torch.testing.assert_close(
        extras["log"]["Episode_Termination/time_out"],
        torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
    )


def test_single_reward_group_uses_standard_z_rl_metric_names() -> None:
    """Keep the common one-group case aligned with IsaacLab/Z-RL log names."""
    env = ActiveAdaptationVecEnvWrapper(
        _FakeAAEnv(),
        observation_keys=("policy",),
        reward_keys=("loco",),
        include_episode_stats=True,
    )

    _observations, _rewards, _dones, extras = env.step(torch.zeros(4, 2))

    assert "Episode_Reward/track_velocity" in extras["log"]
    assert "Episode_Reward/loco/track_velocity" not in extras["log"]


def test_wrapper_warns_once_for_unsupported_aa_discount() -> None:
    """Warn once when AA emits discounts unsupported by current Z-RL PPO."""
    env = _make_wrapper(_FakeAAEnv(discount=0.8), discount_mode="warn")

    with pytest.warns(RuntimeWarning, match="non-unit per-step discount"):
        env.step(torch.zeros(4, 2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env.step(torch.zeros(4, 2))
    assert caught == []


def test_ignore_discount_mode_does_not_inspect_gpu_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the default discount path free of value-dependent synchronization."""
    env = _make_wrapper(_FakeAAEnv(discount=0.8))

    def fail_allclose(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise AssertionError("ignore mode must not inspect discount values")

    monkeypatch.setattr(torch, "allclose", fail_allclose)
    _observations, _rewards, _dones, extras = env.step(torch.zeros(4, 2))

    torch.testing.assert_close(extras["discount"], torch.full((4,), 0.8))


def test_wrapper_can_reject_non_unit_aa_discount() -> None:
    """Reject unsupported AA discounts when strict handling is requested."""
    env = _make_wrapper(_FakeAAEnv(discount=0.8), discount_mode="error")

    with pytest.raises(RuntimeError, match="does not consume it"):
        env.step(torch.zeros(4, 2))


def test_wrapper_validates_action_shape() -> None:
    """Reject flat actions that do not match AA's configured action input."""
    env = _make_wrapper()

    with pytest.raises(ValueError, match="Expected actions with shape"):
        env.step(torch.zeros(4, 3))


def test_current_z_rl_ppo_runner_trains_through_wrapper() -> None:
    """Collect and update current Z-RL PPO through the AA wrapper."""
    env = ActiveAdaptationVecEnvWrapper(
        _FakeAAEnv(),
        observation_keys=("policy",),
        reward_keys=("loco",),
    )
    runner = OnPolicyRunner(env, _train_config(), log_dir=None, device="cpu")
    runner.learn(num_learning_iterations=1)

    assert runner.current_learning_iteration == 0
    assert env.env.step_calls == 2


def test_current_z_rl_runner_prints_aa_episode_stats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise AA stats through the real Z-RL logger and console formatter."""
    env = ActiveAdaptationVecEnvWrapper(
        _FakeAAEnv(),
        observation_keys=("policy",),
        reward_keys=("loco",),
        include_episode_stats=True,
    )
    train_cfg = _train_config()
    train_cfg["run_name"] = "aa-rich-log-smoke"

    runner = OnPolicyRunner(env, train_cfg, log_dir=str(tmp_path), device="cpu")
    runner.learn(num_learning_iterations=1)

    output = capsys.readouterr().out
    assert "Run name: aa-rich-log-smoke" in output
    assert "Episode_Reward/track_velocity:" in output
    assert "Episode_Termination/time_out:" in output
