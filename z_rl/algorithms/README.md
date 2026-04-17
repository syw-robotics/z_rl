# Algorithms

This directory contains the learning algorithms used by Z-RL and the explicit extension API used to add custom PPO
loss terms.

## Overview

The core algorithm classes are:

- `PPO`: on-policy reinforcement learning with actor/critic models and rollout-based updates.
- `Distillation`: behavior-cloning style training where a student model learns to match a teacher model.
- `EncoderEstimationPPO`: `ComposablePPO` preset with encoder-latent estimation enabled.

The preferred explicit extension path is now the composition API:

- `composition/composable_ppo.py`: shared PPO builder plus thin subclass that applies configured loss specs
- `composition/specs.py`: contract for extra optimization or logging terms
- `variants/`: concrete algorithm variants and their variant-specific loss specs

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
- updates observation normalizers during rollout collection
- bootstraps rewards on `time_outs` for infinite-horizon environments
- computes GAE-style returns and advantages
- runs clipped PPO surrogate updates over feedforward or recurrent minibatches
- optionally adapts the learning rate based on KL divergence
- optionally integrates symmetry augmentation / mirror loss through `symmetry_cfg`
- supports multi-GPU gradient averaging through `reduce_parameters()`

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
- updates only the student normalizer and student parameters
- skips return computation because it is not an RL objective
- supports gradient accumulation via `gradient_length`
- supports either `mse` or `huber` behavior loss
- supports multi-GPU gradient averaging for the student model

One important loading behavior:

- when loading an RL checkpoint and `load_cfg` is omitted, the teacher defaults to `actor_state_dict`

## Construction Flow

Both algorithms expose `construct_algorithm(obs, env, cfg, device)` as the main config-driven entry point.

Shared behavior:

- resolve classes from `class_name`
- resolve observation groups from `cfg["obs_groups"]`
- inject wrapper-provided observation time-slice metadata into model configs when supported
- create `RolloutStorage`
- instantiate the algorithm class

Algorithm-specific behavior:

- `PPO` resolves symmetry configuration and may share CNN encoders from actor to critic
- `Distillation` forbids symmetry extensions and builds separate `student` / `teacher` models

## PPO Extension Design

`ComposablePPO` is the intended extension point for algorithm-side customization.

Preferred extension is explicit:

```python
from z_rl.algorithms.composition import ComposablePPO, PPOLossSpec


class MyPPO(ComposablePPO):
    @classmethod
    def build_loss_spec(cls, env, algorithm_cfg) -> PPOLossSpec:
        return MyAuxLossSpec(...)
```

The variant-owned loss spec implements:

- `validate(algo)`
- `compute(algo, minibatch)`

Return:

- `opt_losses`: extra loss terms added to the optimization objective
- `non_opt_losses`: metrics that are logged but not optimized

Important contract:

- keys in `opt_losses` are weighted in `PPO.update()` by an attribute named `<key>_coef` if it exists
- custom loss keys must not collide with base PPO keys such as `surrogate_loss`, `value_loss`, or `entropy`

This keeps the extension surface local to the actual customization and avoids requiring users to reason about multiple
inheritance order.

When creating a new PPO variant, the intended workflow is:

- implement a `PPOLossSpec`
- subclass `ComposablePPO`
- override `build_loss_spec(env, algorithm_cfg)`
- keep variant-specific spec classes in the same file as the variant unless they are shared across multiple variants

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
- `distillation.py`: student/teacher distillation loop with gradient accumulation

## Maintenance Notes

- Keep runner-facing method names aligned across algorithms so config-driven construction stays predictable.
- Prefer extending PPO through `ComposablePPO` and `PPOLossSpec` instead of copying the whole class.
- If algorithm checkpoint keys change, update both this README and any runner or plugin template code that depends on
  them.
