# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from pathlib import Path
from tensordict import TensorDict
from types import SimpleNamespace
from typing import ClassVar

from z_rl.algorithms.flash_sac import _project_distribution
from z_rl.runners import OnPolicyRunner


class _VectorEnv:
    num_envs, num_actions, device = 4, 2, "cpu"
    cfg: ClassVar[dict] = {}
    single_action_space = SimpleNamespace(low=[-2.0, 0.5], high=[1.0, 4.5])

    def __init__(self) -> None:
        self.obs = TensorDict({"policy": torch.zeros(4, 2), "priv": torch.arange(4.0)[:, None]}, [4])
        self.steps = 0

    def get_observations(self) -> TensorDict:
        return self.obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert actions.device == self.obs["policy"].device
        assert (
            (actions >= torch.tensor(self.single_action_space.low))
            & (actions <= torch.tensor(self.single_action_space.high))
        ).all()
        self.obs["policy"].add_(1)
        terminal = self.obs.clone()
        terminal["policy"].fill_(99)
        self.steps += 1
        return (
            self.obs,
            -actions.square().sum(-1),
            torch.tensor([True, True, True, False]),
            {
                "time_outs": torch.tensor([True, True, False, False]),
                "terminated": torch.tensor([False, True, True, False]),
                **({"time_outs_obs": terminal} if self.steps == 1 else {}),
            },
        )


def _runner() -> OnPolicyRunner:
    cfg = {
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "priv"]},
        "num_steps_per_env": 1,
        "save_interval": 10,
        "algorithm": {
            "class_name": "z_rl.algorithms.flash_sac:FlashSAC",
            "batch_size": 8,
            "actor_hidden_dim": 8,
            "critic_hidden_dim": 8,
            "learning_starts": 8,
            "replay_buffer_size": 32,
        },
    }
    return OnPolicyRunner(_VectorEnv(), cfg)


def test_flash_sac_replay_checkpoint_and_policy(tmp_path: Path) -> None:
    """Train through the runner and resume with the same deterministic policy."""
    runner = _runner()
    alg = runner.alg
    before = alg.actor.mean.weight.detach().clone()
    runner.learn(1)
    assert alg.update_step == 0
    replay = alg.replay.data
    torch.testing.assert_close(replay["bootstrap"][0], torch.tensor([True, False, False, True]))
    torch.testing.assert_close(replay["obs"]["policy"][0], torch.zeros(4, 2))
    torch.testing.assert_close(replay["next_obs"]["policy"][0, :, 0], torch.tensor([99.0, 1.0, 1.0, 1.0]))
    assert replay["actions"][0].abs().max() <= 1

    runner.learn(2)
    assert alg.update_step == 2
    torch.testing.assert_close(replay["bootstrap"][1], torch.tensor([False, False, False, True]))
    assert not torch.equal(alg.actor.mean.weight, before)
    obs = runner.env.get_observations()
    policy = runner.get_inference_policy()
    expected = policy(obs)
    torch.testing.assert_close(policy.as_onnx()(obs["policy"]), expected)

    checkpoint = tmp_path / "flash_sac.pt"
    runner.save(str(checkpoint), infos={"algorithm": "flash_sac"})
    restored = _runner()
    assert restored.load(str(checkpoint)) == {"algorithm": "flash_sac"}
    torch.testing.assert_close(restored.get_inference_policy()(obs), expected)
    torch.testing.assert_close(restored.alg.target_critic.state_dict(), alg.target_critic.state_dict())
    torch.testing.assert_close(restored.alg.return_rms.state_dict(), alg.return_rms.state_dict())
    torch.testing.assert_close(restored.alg.actor_optimizer.state_dict(), alg.actor_optimizer.state_dict())
    assert restored.alg.replay.size == 0
    restored.learn(2)
    assert restored.alg.update_step == 3


def test_flash_sac_projection_preserves_mass_at_terminal_and_saturated_bins() -> None:
    """Retain probability mass at the bin boundaries and for terminal rewards."""
    support = torch.tensor([-1.0, 0.0, 1.0])
    probs = torch.tensor([[0.3, 0.2, 0.5]]).expand(3, -1)
    reward = torch.tensor([0.25, 2.0, 0.0])
    discount = torch.tensor([0.0, 0.5, 1.0])
    targets = _project_distribution(probs, reward, discount, torch.zeros(3), support)
    torch.testing.assert_close(targets, torch.tensor([[0.0, 0.75, 0.25], [0.0, 0.0, 1.0], [0.3, 0.2, 0.5]]))
