---
name: z_rl
description: Modular RL framework for IsaacLab; three Spec abstractions (Latent/Head/PPOLoss) for composable models and PPO losses.
---

# z_rl

Modular RL framework built on rsl_rl for IsaacLab robot training. Composability via three Spec abstractions.

## Three Core Specs

- `LatentSpec` (`z_rl/models/composition/specs.py:8`) — replaces obs→latent adapter in `MLPModel`. Implement `build(model) -> nn.Module`, `get_latent_dim(model) -> int`. Built-in: `CNNLatentSpec`.
- `HeadSpec` (`z_rl/models/composition/specs.py:36`) — replaces output head. Implement `build(model, input_dim, output_dim, activation) -> nn.Module`. Built-in: `SimBaHeadSpec`.
- `PPOLossSpec` (`z_rl/algorithms/composition/specs.py:10`) — injects extra loss into PPO. Implement `compute(algo, minibatch) -> (opt_losses, non_opt_losses)`. Fields named `<key>_coef` auto-copy onto algo. Built-ins: `EncoderEstimationLossSpec`, `MoERoutingLossSpec`.

## Composition Entry Points

- `ComposableModel` (`z_rl/models/composition/composable_model.py:24`) — `MLPModel` subclass accepting `latent_spec` / `head_spec`.
- `ComposablePPO` (`z_rl/algorithms/composition/composable_ppo.py:31`) — `PPO` subclass accepting `loss_spec`.
- `PPO.construct_algorithm` (`z_rl/algorithms/ppo.py:445`) — unified build entry.

## Directory Layout

- `models/composition/` — LatentSpec, HeadSpec, ComposableModel, adapters
- `models/variants/` — cnn, rnn, simba, moe, mlp_encoder, group_mlp_encoder model implementations
- `algorithms/composition/` — PPOLossSpec, ComposablePPO
- `algorithms/variants/` — encoder_estimation_ppo, moe_ppo
- `modules/` — nn primitives (mlp, cnn, rnn, moe, simba, vae, distribution)
- `runners/` — on_policy_runner, distillation_runner
- `adaptor/isaaclab/rl_cfg.py` — `@configclass` hierarchy for IsaacLab (ZRlMLPModelCfg → ZRlComposableModelCfg → variants; ZRlPpoAlgorithmCfg → ZRlComposablePpoAlgorithmCfg → variants)
- `cli/` — z-rl-plugin-init, z-rl-checkpoint-editor
- `utils/` — resolve_spec, bind_matching_fields, ObsSelector

## Adding a New Spec

- **LatentSpec**: subclass in `models/variants/`, implement `build()` + `get_latent_dim()`. Use via `ComposableModel(latent_spec={"class_name": "YourSpec", ...})` or set `latent_spec_class` on a named subclass.
- **HeadSpec**: subclass, implement `build(model, input_dim, output_dim, activation)`. Same wiring via `head_spec`.
- **PPOLossSpec**: subclass, implement `compute()`. Add `<key>_coef` fields. Use via `ComposablePPO(loss_spec={"class_name": "YourSpec", ...})`.

## Key Utilities

- `resolve_spec(ref)` — resolves spec from class / string / dict with `class_name` / instance.
- `bind_matching_fields(spec, kwargs)` — copies kwargs matching spec `__init__` params onto the spec (flat configs).
- `ObsSelector` — selects obs groups from TensorDict by name.
