# Active Adaptation Adaptor for Z-RL

This adaptor lets the current Z-RL on-policy runner use an [Active Adaptation
(AA)](https://github.com/Agent-3154/active-adaptation) environment without constructing or importing an AA learning policy. It
wraps AA above its backend-independent TorchRL environment boundary, so the same
wrapper applies to IsaacLab, Mjlab, Mujoco, and Motrix environments supported by AA.

## Scope

This early version supports:

- Z-RL's flat continuous action interface with one named AA action input
- explicit AA observation-group selection
- explicit aggregation of one or more AA reward groups
- AA automatic reset, termination, truncation, and optional detailed episode statistics
- pass-through of AA's per-step discount in `extras["discount"]`
- lazy PPO symmetry augmentation and mirror loss for dense vector observation groups

Current Z-RL PPO does not consume AA's per-step discount. `discount_mode="ignore"`
continues with Z-RL's done-based return semantics without synchronizing the GPU
to inspect discounts. Use `discount_mode="warn"` while validating a task, or
`discount_mode="error"` to reject non-unit discounts.

## Usage

AA must be initialized before constructing the environment because simulator
startup and backend registration are process-scoped.

```python
import active_adaptation as aa

from z_rl.adaptor.active_adaptation import (
    ActiveAdaptationVecEnvWrapper,
    make_active_adaptation_env,
)
from z_rl.runners import OnPolicyRunner

aa.init(cfg, auto_rank=True)

aa_env = make_active_adaptation_env(
    cfg.task,
    device=cfg.device,
    headless=cfg.headless,
    seed=cfg.seed,
)
env = ActiveAdaptationVecEnvWrapper(
    aa_env,
    observation_keys=("policy",),
    reward_keys=("loco",),
    action_key="action",
    discount_mode="ignore",
    include_episode_stats=True,
)

runner = OnPolicyRunner(
    env,
    train_cfg=z_rl_cfg,
    log_dir=log_dir,
    device=cfg.device,
)
runner.learn(num_learning_iterations=z_rl_cfg["max_iterations"])
```

Map Z-RL actor and critic inputs using its existing `obs_groups` configuration:

```python
z_rl_cfg["obs_groups"] = {
    "actor": ["policy"],
    "critic": ["policy", "priv"],
}
```

Enable symmetry augmentation through the standard Z-RL PPO configuration. The
adaptor compiles the selected AA observation-group transforms and configured
action-input transform once, then appends mirrored samples only while updating
PPO mini-batches.

```python
z_rl_cfg["algorithm"]["symmetry_augmentation"] = True
z_rl_cfg["algorithm"]["symmetry_cfg"] = {
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.1,
}
```

Every selected observation group and the action input must provide an AA
`SymmetryTransform`. The current adaptor supports dense tensors shaped
`(num_envs, features)`; transforms are checked for matching width and for being
involutive signed permutations before training starts.

With episode statistics enabled, AA reward terms are logged as
`Episode_Reward/<term>` and termination terms as
`Episode_Termination/<term>`. Reward terms are divided by AA's configured
maximum episode length for the same per-step scale used by IsaacLab logs. The
wrapper leaves stats on the environment device; the Z-RL logger reuses its
completed-environment indices, so this path adds no device-host synchronization.

`make_active_adaptation_env()` deliberately uses AA's `_EnvBase.registry` because
AA currently has no public environment-only factory. Pin the AA version and run
the adaptor contract tests when updating AA.

## TODO

- Consume AA's per-step discount in Z-RL GAE for objective parity with AA PPO.
- Coordinate AA and Z-RL distributed initialization for multi-GPU training.
- Materialize and mirror AA functional observation groups.
- Support image/channel symmetry and other non-vector observation layouts.
