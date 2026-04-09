# IsaacLab Adaptor for Z-RL

This package adapts IsaacLab `ManagerBasedRLEnv` environments to the interfaces expected by Z-RL.

It currently provides:

- `ZRlVecEnvWrapper`: converts IsaacLab env outputs to `z_rl.env.VecEnv`
- `ZRlBaseRunnerCfg` / `ZRlOnPolicyRunnerCfg` and related config classes in `rl_cfg.py`
- `ZRlDistillationRunnerCfg` and distillation algorithm config in `distillation_cfg.py`
- `ZRlSymmetryCfg` for symmetry augmentation settings
- `export_policy_as_jit()` / `export_policy_as_onnx()` for policy export

## Quick start

```python
from isaaclab_rl.z_rl import ZRlVecEnvWrapper

env = ZRlVecEnvWrapper(
    env,
    clip_actions=1.0,
    obs_group_concat_mode="term_major",  # or "history_major"
)
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
- caches observation metadata such as `obs_format`, `obs_group_layout_mode_map`, and time-slice selectors

## Observation groups and `concatenate_terms`

In IsaacLab, each observation group is controlled by `ObservationGroupCfg.concatenate_terms`.

- `concatenate_terms=True`: IsaacLab already returns one tensor for that group
- `concatenate_terms=False`: IsaacLab returns a term dictionary, for example:

```python
{
    "policy": {
        "base_lin_vel": ...,
        "joint_pos": ...,
    }
}
```

`ZRlVecEnvWrapper` only needs to re-layout groups that come back as dictionaries. This is the key relationship:

- if `concatenate_terms=True`, the wrapper does not reorder that group and `obs_group_concat_mode` has no effect on it
- if `concatenate_terms=False`, the wrapper converts the dict group into a single tensor for Z-RL

So, `obs_group_concat_mode` only matters for groups whose IsaacLab config sets `concatenate_terms=False`.

## `term_major` vs `history_major`

This is the most important part of the wrapper.

Assume one observation group contains two terms:

- `a`: single-frame dim `Da`
- `b`: single-frame dim `Db`
- both have history length `H`

### `term_major`

`term_major` keeps IsaacLab's default per-term history layout.

Conceptually, the flattened order is:

```text
[a_t-H+1, a_t-H+2, ..., a_t, b_t-H+1, b_t-H+2, ..., b_t]
```

If each term is already flattened by IsaacLab to shape `(N, H * D)`, wrapper concatenation produces:

```text
(N, H*Da + H*Db)
```

This is the default mode and also the fallback mode.

### `history_major`

`history_major` first restores each term into `(N, H, D)` and then concatenates features inside the same time step.

Conceptually, the flattened order becomes:

```text
[a_t-H+1, b_t-H+1, a_t-H+2, b_t-H+2, ..., a_t, b_t]
```

The final shape is:

```text
(N, H * (Da + Db))
```

This layout is useful when downstream code wants to slice "the last frame" or "all but the last frame" as contiguous blocks.

## Exact relationship between `history_major` and `self.concatenate_terms`

To actually get `history_major` for one observation group, all of the following must be true:

1. In IsaacLab, that group must use `self.concatenate_terms = False`
2. When creating the wrapper, `obs_group_concat_mode="history_major"`
3. The group must be time-slice compatible:
   - every term has the same non-zero history length
   - every term is a vector per frame
   - the group is returned as a dict, not already concatenated by IsaacLab

If any of these conditions is not satisfied, the wrapper uses `term_major` for that group.

That means:

- `self.concatenate_terms=True` and `obs_group_concat_mode="history_major"` is not an error, but that group will still stay in IsaacLab's existing layout
- `self.concatenate_terms=False` is necessary, but not sufficient; the term shapes and history lengths must also be compatible

## Practical examples

### Example 1: IsaacLab already concatenates terms

IsaacLab config:

```python
self.concatenate_terms = True
```

Wrapper config:

```python
ZRlVecEnvWrapper(env, obs_group_concat_mode="history_major")
```

Result:

- the group is already a tensor when it reaches the wrapper
- the wrapper does not rebuild its layout
- effective layout remains IsaacLab's original concatenation

### Example 2: Dict group with default wrapper behavior

IsaacLab config:

```python
self.concatenate_terms = False
```

Wrapper config:

```python
ZRlVecEnvWrapper(env, obs_group_concat_mode="term_major")
```

Result:

- the wrapper concatenates terms in term order
- history stays grouped inside each term

### Example 3: Dict group with history-major layout

IsaacLab config:

```python
self.concatenate_terms = False
```

Wrapper config:

```python
ZRlVecEnvWrapper(env, obs_group_concat_mode="history_major")
```

Compatible group result:

- each term is viewed as `(N, H, D)`
- terms are concatenated along feature dim inside each time step
- output is flattened to `(N, H * sum(D_i))`

Incompatible group result:

- the wrapper silently falls back to `term_major` for that group

## Time slicing with `get_obs_time_slice()`

The wrapper can precompute efficient selectors for compatible groups:

- `"last"`
- `"exclude_last"`
- `"exclude_first"`

This is exposed through:

```python
last_obs = env.get_obs_time_slice(obs["policy"], "policy", "last")
```

This is only available for groups where:

- all terms share the same non-zero history length
- each single frame is a vector

For `history_major` groups, these slices are contiguous ranges.
For compatible `term_major` groups, they may be index tensors because the latest frame pieces are interleaved by term block rather than stored contiguously.

## Cached metadata exposed by the wrapper

Useful properties:

- `obs_format`
  - structure: `{group_name: {term_name: (history_length, *single_frame_shape)}}`
- `obs_group_layout_mode_map`
  - actual layout used per group: `"term_major"` or `"history_major"`
- `obs_group_time_slice_map`
  - cached selectors used by `get_obs_time_slice()`

These properties are useful when model code needs to know how a flattened vector was assembled.

## Recommended configuration

Use `term_major` when:

- you want the default IsaacLab-compatible layout
- some groups are already concatenated in IsaacLab
- history lengths differ across terms

Use `history_major` when:

- downstream code operates on frame-wise windows
- you want cheap access to the latest frame or shifted history blocks
- your target groups are configured with `concatenate_terms=False`
- those groups contain vector terms with the same non-zero history length

## File overview

- `vecenv_wrapper.py`: core IsaacLab-to-Z-RL environment wrapper
- `rl_cfg.py`: training runner, algorithm, and model config classes
- `distillation_cfg.py`: distillation-specific runner and algorithm configs
- `symmetry_cfg.py`: symmetry augmentation config
- `exporter.py`: TorchScript / ONNX export helpers
