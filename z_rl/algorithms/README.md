# Algorithms

This directory contains the learning algorithms used by Z-RL and the explicit extension API used to add custom PPO
loss terms.

## Overview

The core algorithm classes are:

- `PPO`: on-policy reinforcement learning with actor/critic models and rollout-based updates.
- `SAC`: off-policy continuous control with a squashed Gaussian actor and twin action-value networks.
- `Distillation`: behavior-cloning style training where a student model learns to match a teacher model.
- `EncoderEstimationPPO`: `ComposablePPO` preset with encoder-latent estimation enabled.
- `MoEPPO`: `ComposablePPO` preset with MoE gate-entropy and expert-balance regularizers.

The preferred explicit extension path is now the composition API:

- `composition/composable_ppo.py`: PPO that installs a `loss_spec` from config or `loss_spec_class`
- `composition/specs.py`: contract for extra optimization or logging terms
- `variants/`: named presets (`MoEPPO`, `EncoderEstimationPPO`) and their loss specs

## Current Structure

Both algorithms follow the same runner-facing lifecycle:

1. `act(obs)`: run policy inference and cache transition data
2. `process_env_step(obs, rewards, dones, extras)`: record one environment step
3. `compute_returns(obs)`: prepare training targets if the algorithm needs them
4. `update()`: consume storage and run optimization

Both classes also provide:

- `train_mode()` / `eval_mode()`
- `save()` / `load(...)`
- `get_policy()`
- `construct_algorithm(...)`

## PPO

`PPO` owns two learnable models:

- `actor`: action policy
- `critic`: value function

At a high level, one PPO iteration looks like this:

```text
obs -> actor/critic inference -> rollout storage -> GAE returns -> minibatch loss -> gradient step
```

Main responsibilities of `PPO`:

- samples actions with the actor and records log-probabilities and value estimates
- bootstraps rewards on `time_outs` for infinite-horizon environments
- computes GAE-style returns and advantages
- runs clipped PPO surrogate updates over feedforward or recurrent minibatches
- updates observation normalizers once per learning iteration after the optimizer step, using stored rollout observations
- optionally runs the forward pass and loss computation in bfloat16 mixed precision
- optionally adapts the learning rate based on KL divergence
- reports policy KL, PPO clip fraction, and critic explained variance
- optionally integrates symmetry augmentation / mirror loss through `symmetry_cfg`
- supports multi-GPU gradient averaging through `reduce_parameters()`
- keeps `class_name` and other constructor keys in the original training config so loggers can record them

## SAC

Select `ZRlSacRunnerCfg` in an IsaacLab task's agent configuration. It uses the existing
`OnPolicyRunner` collection, logging, checkpoint, and ONNX interfaces. Each SAC update samples replay.

```python
from z_rl.adaptor.isaaclab import ZRlSacRunnerCfg, ZRlVecEnvWrapper
from z_rl.runners import OnPolicyRunner

cfg = ZRlSacRunnerCfg(
    experiment_name="sac", max_iterations=1000, save_interval=100,
    obs_groups={"actor": ["policy"], "critic": ["policy"]},
)
env = ZRlVecEnvWrapper(env, clip_actions=cfg.clip_actions)
runner = OnPolicyRunner(env, cfg.to_dict(), log_dir="logs/sac", device=env.device)
runner.learn(cfg.max_iterations)
```

- `replay_buffer_size` and `learning_starts` count individual transitions per training process.
  Replay holds whole vector steps and samples uniformly with replacement. Each iteration collects
  `num_steps_per_env` vector steps, then runs `updates_per_iteration` critic updates of `batch_size` samples.
  The actor updates every `policy_frequency` critic steps. Set `auto_alpha=False` for a fixed temperature.
- `n_steps` sets the return horizon. Returns stop at the first episode boundary. For infinite-horizon
  tasks on IsaacLab versions with `env_cfg.compute_final_obs`, set it to `True` before creating the
  environment. The wrapper converts `final_obs` into `time_outs_obs`. Active Adaptation supplies this
  observation from its pre-reset transition. A timeout without a snapshot gets a reward-only final target.
- The actor uses finite Box bounds and wrapper clipping in environment action units. Unbounded dimensions
  default to `[-1, 1]`. To use task-specific limits, set `cfg.actor.distribution_cfg.action_low` and
  `action_high` to scalars or lists in action order. Inference and ONNX apply the same bounds.
- Checkpoints contain both Q networks, their targets, actor, normalizers, optimizers, temperature, and
  update counter. A resumed run fills a fresh replay buffer and observes the same warm-up threshold.

The SAC update follows the [RSL-RL-SAC reference implementation](https://github.com/leggedrobotics/rsl_rl_sac).

## FlashSAC

Use `ZRlFlashSacRunnerCfg` for the single-device FlashSAC baseline. The preset collects one vector step per
iteration and uses a one-step categorical Bellman target with a fixed learning rate. It accepts flat actor and critic
observation groups. By default, the actor uses `policy` and the critic uses `critic` when available or `policy`
otherwise. Set both groups explicitly in a task preset.

```python
from z_rl.adaptor.isaaclab import ZRlFlashSacRunnerCfg, ZRlVecEnvWrapper
from z_rl.runners import OnPolicyRunner

cfg = ZRlFlashSacRunnerCfg(max_iterations=10_000, obs_groups={"actor": ["policy"], "critic": ["policy"]})
env = ZRlVecEnvWrapper(env, clip_actions=cfg.clip_actions)
runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=log_dir, device=cfg.device)
runner.learn(cfg.max_iterations)
```

`replay_buffer_size` and `learning_starts` count individual transitions across environments. The policy uses uniform
actions during replay warm-up. Checkpoints restore the networks, optimizers, and reward statistics. Resumed training
rebuilds replay from the environment and repeats warm-up. Timeout bootstrapping requires a pre-reset observation in
`extras["time_outs_obs"]`.

## RLPD

Both presets train the SAC actor from live environment replay and a fixed prior rollout. Use the same
environment wrapper, observation groups, runner and export calls as in the SAC example above.

| IsaacLab runner config | Minibatch when `batch_size=256` | Critic and actor updates per vector step at defaults |
| --- | --- | --- |
| `ZRlRlpdRunnerCfg` | 128 online + 128 prior | Train 10 LayerNorm Q heads for 20 updates. Each target uses the minimum of two randomly selected target heads. Update the actor and temperature once using the ensemble mean for the actor objective. |
| `ZRlBtxRlpdRunnerCfg` | 256 online + 102 prior | Train twin scalar Q heads for four updates using three-step returns. Update the actor and temperature once, with return-gated behavior cloning on prior actions. |

```python
from z_rl.adaptor.isaaclab import ZRlRlpdRunnerCfg, ZRlBtxRlpdRunnerCfg

cfg = ZRlRlpdRunnerCfg(  # use ZRlBtxRlpdRunnerCfg for gated BC
    experiment_name="rlpd", max_iterations=1000, save_interval=100,
    obs_groups={"actor": ["policy"], "critic": ["policy"]},
)
cfg.algorithm.prior_data = "/path/to/rollout_400_8.pt"
```

For `RLPD`, `prior_data_ratio` is the fraction of the total batch (default 0.5). For `BTXRLPD`,
it is the number of prior samples per online sample (default 0.4). Both round the prior count down.
`updates_per_iteration` counts critic gradient steps after collecting `num_steps_per_env` vector steps.
`policy_frequency=None` updates the actor and temperature after the last critic step of each iteration.
The ensemble and minibatch design follow the [original RLPD implementation](https://github.com/ikostrikov/rlpd).

The BTX preset ports the prior-data and gated-BC objectives onto Z-RL's scalar SAC critic. It averages
four sampled actions per state and excludes episode-initial rows from actor and critic losses.
For a prior row with more than `1 / (1 - gamma)` recorded steps remaining, the BC weight is
`clamp((1 - gamma) * (return_to_go - policy_Q), 0, 1)`. Both values use the same reward units.
Set `bc_loss="mse"` (default coefficient 1.0) or `bc_loss="nll"` (default 0.05).
`bc_coef=0` disables BC while keeping the mixed replay updates.

**Prior archive.** Pass a trusted `.pt` file produced by Active Adaptation's `scripts/rollout.py` or
its relabel stage. Format version 1 stores a `stacked` TensorDict with batch dimensions
`[time, environment]`. Observation group names, feature order and action units must match the live task.
The loader reads the action and reward keys selected by the Active Adaptation wrapper. With a generic
VecEnv it reads `action` and sums the components of `next/reward`, including grouped rewards.
Full archives use the recorded pre-reset `next` observations at timeouts. Compact archives use
adjacent rows within each environment, omit the final row as a transition, and stop bootstrapping at
resets. Both forms honor per-step discounts. Training samples the prior on CPU and transfers minibatches
to the training device. Resuming reloads the prior from the configured path and warms up fresh online
replay. Inference and ONNX export require only the checkpoint and model configuration.

## Distillation

`Distillation` owns two models with different roles:

- `student`: the learnable policy returned by `get_policy()`
- `teacher`: a frozen or preloaded reference policy used as supervision

The distillation loop is:

```text
obs -> student action + teacher action -> storage -> behavior loss -> gradient accumulation -> optimizer step
```

Main responsibilities of `Distillation`:

- records student actions and teacher target actions during rollout collection
- updates only the student normalizer (once per learning iteration after the optimizer step) and student parameters
- skips return computation because it is not an RL objective
- supports gradient accumulation via `gradient_length`
- warns when `num_learning_epochs * num_steps_per_env` is not divisible by `gradient_length`
- optionally runs the forward pass and loss computation in bfloat16 mixed precision
- supports either `mse` or `huber` behavior loss
- supports multi-GPU gradient averaging for the student model
- keeps `class_name` and other constructor keys in the original training config so loggers can record them

One important loading behavior:

- when loading an RL checkpoint and `load_cfg` is omitted, the teacher defaults to `actor_state_dict`
- checkpoints also store `student_modules` and `teacher_modules` (PPO: `actor_modules`, `critic_modules`). `z_rl.utils.checkpoint` packs and reads those entries. `load_cfg["student_modules"]` maps a student child name to a source role, for example `{"head": "actor"}`, and copies that child after any full student load

## Construction Flow

Both algorithms expose `construct_algorithm(obs, env, cfg, device)` as the main config-driven entry point.

Shared behavior:

- resolve classes from `class_name`
- resolve observation groups from `cfg["obs_groups"]`
- inject wrapper-provided observation time-slice metadata into model configs when supported
- create `RolloutStorage`
- instantiate the algorithm class

Algorithm-specific behavior:

- `PPO` resolves symmetry configuration
- `Distillation` forbids symmetry extensions and builds separate `student` / `teacher` models

## PPO Extension Design

`ComposablePPO` is the intended extension point for algorithm-side customization.

The user-facing unit is the loss spec. Config can point `ComposablePPO` at a spec without a named algorithm subclass:

```python
algorithm = {
    "class_name": "ComposablePPO",
    "loss_spec": {
        "class_name": "MyAuxLossSpec",
        "my_aux_loss_coef": 0.1,
    },
}
```

```python
from dataclasses import dataclass

from z_rl.algorithms.composition import PPOLossSpec


@dataclass
class MyAuxLossSpec(PPOLossSpec):
    my_aux_loss_coef: float = 0.1

    def compute(self, algo, minibatch):
        return {"my_aux_loss": ...}, {}
```

`PPOLossSpec` requires `compute(algo, minibatch)`. Optional:

- `validate(algo)`: no-op by default
- fields named `<loss_key>_coef`: copied onto the algorithm so `PPO.update()` can weight that extra loss

Return:

- `opt_losses`: extra loss terms added to the optimization objective
- `non_opt_losses`: metrics that are logged but not optimized

Important contract:

- keys in `opt_losses` are weighted in `PPO.update()` by an attribute named `<key>_coef` if it exists
- custom loss keys must not collide with base PPO keys such as `surrogate_loss`, `value_loss`, or `entropy`

Named presets (`MoEPPO`, `EncoderEstimationPPO`) set `loss_spec_class` so existing `class_name` values keep working.
Subclass `ComposablePPO` only when the spec cannot express the behavior, for example a custom `act()`.

The shared `ComposablePPO.construct_algorithm(...)` builder handles actor/critic/storage assembly so variants do not
need to duplicate PPO construction boilerplate.

If a PPO variant needs custom rollout actions, override `act(obs)` in the `ComposablePPO` subclass. The inherited
implementation uses the standard PPO rollout behavior: sample `actor(obs, stochastic_output=True)`, evaluate the
critic, and record the matching log probability and distribution parameters.

## Reference Implementations

Use these files as the canonical examples:

- `ppo.py`: base PPO training loop, KL scheduling, symmetry integration, and checkpoint I/O
- `composition/composable_ppo.py`: shared composition builder for PPO variants
- `composition/specs.py`: contract for PPO loss extensions
- `variants/encoder_estimation_ppo.py`: predefined algorithm variant and its `EncoderEstimationLossSpec`
- `variants/moe_ppo.py`: predefined algorithm variant and its `MoERoutingLossSpec`
- `distillation.py`: student/teacher distillation loop with gradient accumulation

## Maintenance Notes

- Keep runner-facing method names aligned across algorithms so config-driven construction stays predictable.
- Prefer extending PPO by writing a `PPOLossSpec` and pointing `ComposablePPO` at it instead of copying the whole class.
- If algorithm checkpoint keys change, update both this README and any runner or plugin template code that depends on
  them.
