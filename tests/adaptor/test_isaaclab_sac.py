from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def test_isaaclab_sac_consumes_final_observations_for_one_step():
    managers = pytest.importorskip("isaaclab.managers")
    from isaaclab.envs import ManagerBasedRLEnv

    from z_rl.adaptor.isaaclab import ZRlSacRunnerCfg, ZRlVecEnvWrapper
    from z_rl.runners import OnPolicyRunner

    env = Mock(spec=ManagerBasedRLEnv)
    env.unwrapped = env
    env.device, env.num_envs, env.max_episode_length = "cpu", 2, 5
    env.action_manager = Mock(total_action_dim=1)
    env.cfg = SimpleNamespace(is_finite_horizon=False)
    env.sim = Mock()
    group = managers.ObservationGroupCfg(concatenate_terms=False)
    group.position = managers.ObservationTermCfg(func=lambda _: torch.zeros(2, 1))
    env.observation_manager = managers.ObservationManager({"policy": group}, env)
    shared_extras = {}
    done = torch.ones(2, dtype=torch.bool)

    def step(_actions):
        shared_extras.pop("final_obs", None)
        timeout = done if env.step.call_count == 1 else ~done
        if timeout.any():
            shared_extras["final_obs"] = {"policy": {"position": torch.full((2, 1), 17.0)}}
        return {"policy": {"position": torch.zeros(2, 1)}}, torch.ones(2), ~done, timeout, shared_extras

    env.step.side_effect = step
    cfg = ZRlSacRunnerCfg(
        experiment_name="test",
        max_iterations=2,
        save_interval=2,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
    )
    cfg.actor.hidden_dims = cfg.critic.hidden_dims = [8]
    cfg.algorithm.learning_starts = cfg.algorithm.replay_buffer_size = 4
    cfg.validate()
    runner = OnPolicyRunner(ZRlVecEnvWrapper(env, clip_actions=0.25), cfg.to_dict())
    runner.learn(2)
    replay = runner.alg.replay
    assert runner.alg.update_step == 1
    torch.testing.assert_close(replay.data["next_obs"]["policy"][0], torch.full((2, 1), 17.0))
    torch.testing.assert_close(replay.data["next_obs"]["policy"][1], torch.zeros(2, 1))
    assert replay.data["bootstrap"].all() and replay.data["actions"].abs().max() <= 0.25
    assert "time_outs_obs" not in shared_extras
