"""Check Z-RL's frame selectors against tensors produced by IsaacLab."""

import torch
from unittest.mock import Mock

import pytest

from z_rl.extensions import TensorMirrorSpec


@pytest.mark.parametrize("newest_first", [False, True])
@pytest.mark.parametrize("order", ["term", "time"])
@pytest.mark.parametrize("concatenate", [True, False])
def test_observation_history(order: str, concatenate: bool, newest_first: bool) -> None:
    """Read current and shifted frames and mirror each frame in its stored layout."""
    managers = pytest.importorskip("isaaclab.managers")
    if not hasattr(managers.ObservationGroupCfg(), "history_order"):
        pytest.skip("IsaacLab does not expose group history ordering")
    from isaaclab.envs import ManagerBasedRLEnv
    from z_rl.adaptor.isaaclab import ZRlVecEnvWrapper

    env = Mock(spec=ManagerBasedRLEnv)
    env.unwrapped = env
    env.num_envs = 1
    env.max_episode_length = 10
    env.device = "cpu"
    env.sim = Mock()
    env.sim.is_playing.return_value = True
    env.action_manager = Mock(total_action_dim=2, active_terms=["joint"])
    action_term = env.action_manager.get_term.return_value
    action_term.action_dim = 2
    action_term.cfg.symmetry_transform.return_value = TensorMirrorSpec([0, 1])
    env.frame = torch.zeros(1, 3)
    group = managers.ObservationGroupCfg(
        concatenate_terms=concatenate, history_order=order, history_length=3, history_newest_first=newest_first
    )
    group.joint = managers.ObservationTermCfg(func=lambda env: env.frame[:, :2])
    group.joint.symmetry_transform = lambda **kwargs: TensorMirrorSpec([1, 0])
    group.clock = managers.ObservationTermCfg(func=lambda env: env.frame[:, 2:])
    group.clock.symmetry_transform = lambda **kwargs: TensorMirrorSpec([0], [-1.0])
    env.observation_manager = managers.ObservationManager({"policy": group}, env)
    wrapper = ZRlVecEnvWrapper(env)
    time_major = concatenate and order == "time"

    frames = torch.arange(9, dtype=torch.float).reshape(3, 3)
    for frame in frames:
        env.frame = frame.unsqueeze(0)
        env.observation_manager.compute(update_history=True)
    observations = wrapper.get_observations()
    assert observations.to(env.device) is observations
    obs = observations["policy"]

    def flatten(window: torch.Tensor) -> torch.Tensor:
        if newest_first:
            window = window.flip((0,))
        if time_major:
            return window.reshape(1, -1)
        return torch.cat((window[:, :2].flatten(), window[:, 2])).unsqueeze(0)

    selectors = wrapper.obs_group_time_slice_map["policy"]
    assert isinstance(selectors["last"].meta, slice) == time_major
    torch.testing.assert_close(obs, flatten(frames))
    torch.testing.assert_close(selectors["last"].select(obs), frames[-1:])
    torch.testing.assert_close(selectors["exclude_first"].select(obs), flatten(frames[1:]))
    torch.testing.assert_close(selectors["exclude_last"].select(obs), flatten(frames[:-1]))

    augmented, _ = wrapper.augment_symmetry(obs=observations)
    mirrored = frames[:, [1, 0, 2]] * torch.tensor([1, 1, -1])
    torch.testing.assert_close(augmented["policy"], torch.cat((obs, flatten(mirrored))))
