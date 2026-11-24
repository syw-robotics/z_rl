"""Read time-major Active Adaptation v1 rollout archives into static CPU replay."""

from pathlib import Path

import torch
from tensordict import TensorDict, TensorDictBase

from .sequence_replay import SequenceReplayBuffer


def _scalar(td, key):
    value = td[key]
    if value.shape == td.batch_size:
        return value
    if value.shape == (*td.batch_size, 1):
        return value.squeeze(-1)
    raise ValueError(f"Expected one {key} per transition, got {tuple(value.shape)}.")


def _returns_to_go(rewards, ends, discounts, gamma):
    returns = torch.empty_like(rewards)
    steps = torch.empty_like(ends, dtype=torch.long)
    future = torch.zeros_like(rewards[0])
    remaining = torch.zeros_like(steps[0])
    for t in reversed(range(len(rewards))):
        future = rewards[t] + gamma * discounts[t] * (~ends[t]) * future
        remaining = 1 + (~ends[t]) * remaining
        returns[t], steps[t] = future, remaining
    return returns, steps


def load_prior_rollout(
    path,
    obs_shapes,
    num_actions,
    *,
    action_key="action",
    reward_keys=None,
    bootstrap_timeouts=True,
    bc_gamma=None,
    n_steps=1,
    include_is_init=False,
):
    """Keep only training fields. Compact archives use the next row of the same environment."""
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "rollout.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"Expected an Active Adaptation format_version=1 rollout at {path}.")
    td = payload.get("stacked")
    if not isinstance(td, TensorDictBase) or td.ndim != 2 or min(td.batch_size) == 0:
        raise ValueError("Expected a nonempty stacked TensorDict with batch dimensions [time, environment].")
    nxt = td["next"]
    obs = td.select(*obs_shapes)
    for key, shape in obs_shapes.items():
        if obs[key].shape[2:] != shape or (key in nxt and nxt[key].shape[2:] != shape):
            raise ValueError(f"Prior observation {key!r} must have per-environment shape {shape}.")
    actions = td[action_key]
    if actions.shape[2:] != (num_actions,):
        raise ValueError(f"Expected {num_actions} prior action components, got {actions.shape[2:]}.")

    reward = nxt["reward"]
    if isinstance(reward, TensorDictBase):
        reward = reward.select(*reward_keys) if reward_keys is not None else reward
        rewards = sum(value.reshape(*td.batch_size, -1).sum(-1) for value in reward.values(True, True))
    else:
        if reward_keys is not None:
            raise ValueError("reward_keys selects groups but the prior contains a reward tensor.")
        rewards = reward.reshape(*td.batch_size, -1).sum(-1)
    rewards = rewards.float()
    done = _scalar(nxt, "done").bool()
    if "is_init" in td:
        is_init = _scalar(td, "is_init").bool()
    else:
        is_init = torch.zeros_like(done)
        is_init[1:] = done[:-1]
    # Relabeled/concatenated trajectories can have an explicit reset with no preceding done.
    reset = torch.zeros_like(done)
    reset[:-1] = is_init[1:] & ~done[:-1]
    ends = done | reset
    bootstrap = ~ends
    env_discount = _scalar(nxt, "discount").float() if "discount" in nxt else torch.ones_like(rewards)
    full_next = all(key in nxt for key in obs_shapes)
    size = len(td) if full_next else len(td) - 1
    if size < n_steps:
        raise ValueError(f"Prior contains {size} usable vector steps, fewer than n_steps={n_steps}.")
    if full_next and bootstrap_timeouts and "truncated" in nxt:
        timeout = done & _scalar(nxt, "truncated").bool()
        if "terminated" in nxt:
            timeout &= ~_scalar(nxt, "terminated").bool()
        bootstrap |= timeout
    next_obs = TensorDict({k: nxt[k][:size] if k in nxt else obs[k][1:] for k in obs_shapes}, [size, td.batch_size[1]])
    data = dict(
        obs=obs[:size],
        next_obs=next_obs,
        actions=actions[:size].float(),
        rewards=rewards[:size],
        segment_end=ends[:size],
        bootstrap=bootstrap[:size],
        env_discount=env_discount[:size],
    )
    if bc_gamma is not None:
        returns, steps = _returns_to_go(rewards, ends, env_discount, bc_gamma)
        data.update(return_to_go=returns[:size], steps_to_go=steps[:size])
    if include_is_init:
        data["is_init"] = is_init[:size]
    replay = SequenceReplayBuffer(size, td.batch_size[1], "cpu")
    replay.data = data
    replay.size = size
    return replay
