# IsaacLab Adaptor for Z-RL

This package adapts IsaacLab `ManagerBasedRLEnv` environments to the interfaces expected by Z-RL.

It currently provides:

- `ZRlVecEnvWrapper`: converts IsaacLab env outputs to `z_rl.env.VecEnv`
- `ZRlBaseRunnerCfg` / `ZRlOnPolicyRunnerCfg` and related config classes in `rl_cfg.py`
- `ZRlDistillationRunnerCfg` and distillation algorithm config in `distillation_cfg.py`
- `ZRlSymmetryCfg` for symmetry augmentation settings
- `export_policy_as_onnx()` for policy export

## Quick start

```python
from z_rl.adaptor.isaaclab import ZRlVecEnvWrapper

env = ZRlVecEnvWrapper(env, clip_actions=1.0)
```

Notes:

- `ZRlVecEnvWrapper` must be the last wrapper in the chain.
- The wrapper calls `env.reset()` during initialization because the Z-RL runner does not call reset first.

## What `ZRlVecEnvWrapper` does

`ZRlVecEnvWrapper` mainly bridges two differences between IsaacLab and Z-RL:

1. Action / step API compatibility
2. Observation layout compatibility

At runtime it:

- exposes `num_envs`, `device`, `max_episode_length`, and `num_actions`
- optionally clips actions before stepping
- converts `(terminated, truncated)` into Z-RL-style `dones`
- adds `extras["time_outs"] = truncated` for infinite-horizon tasks
- returns observations as `TensorDict`
- caches observation metadata such as `obs_format` and time-slice selectors

Those caches metadata enables initializing obs selectors ( as `ObsSelector` objects from `z_rl.utils`), which let downstream code operates observation safer and lazier.

## Observation history layout

Configure the layout on the IsaacLab observation group. For a flat policy vector with one
contiguous block per frame, set the following before creating the environment:

```python
policy = env_cfg.observations.policy
policy.history_length = 5
policy.history_order = "time"
policy.flatten_history_dim = True
policy.concatenate_terms = True
policy.concatenate_dim = -1
```

With two terms and two frames, `"time"` gives `[a0, b0, a1, b1]`. The default
`"term"` gives `[a0, a1, b0, b1]`. The wrapper passes the manager's tensor through and
uses its layout for time slicing and symmetry. Keep the training layout when loading a checkpoint.

For environments that return term dictionaries (`concatenate_terms=False`), the wrapper
concatenates the terms in term order. Use the observation group settings above for frame order.

## Time slicing with cached selectors

The wrapper can precompute efficient selectors for compatible groups:

- `"last"`
- `"exclude_last"`
- `"exclude_first"`

Use the cached selector metadata exposed by the wrapper together with the utility helpers:

```python
from z_rl.utils import resolve_obs_temporal_selector

last_obs_selector = resolve_obs_temporal_selector("policy", "last", env.obs_group_time_slice_map)
last_obs = last_obs_selector.select(obs)
```

`last` selects the newest frame. `exclude_last` drops the newest frame and `exclude_first`
drops the oldest, preserving the stored order of the remaining frames. Frame-major vectors use
contiguous slices. Term-major vectors use cached indices for each term's history.

## Cached metadata exposed by the wrapper

Useful properties:

- `obs_format`
  - structure: `{group_name: {term_name: (history_length, *single_frame_shape)}}`
- `obs_group_time_slice_map`
  - cached selectors used by `resolve_obs_temporal_selector()`

These properties are useful when model code needs to know how a flattened vector was assembled.

## File overview

- `vecenv_wrapper.py`: core IsaacLab-to-Z-RL environment wrapper
- `rl_cfg.py`: training runner, algorithm, and model config classes
- `distillation_cfg.py`: distillation-specific runner and algorithm configs
- `symmetry_cfg.py`: symmetry augmentation config
- `exporter.py`: ONNX export helpers
