# Models

This directory contains the model implementations used by Z-RL and the explicit composition API for latent/head customization.

## Overview

Main models:

- `MLPModel`: base model for vector observations.
- `RNNModel`: recurrent model built on top of the MLP pipeline.
- `CNNModel`: model for mixed 1D/2D observations.
- `ComposableModel`: thin `MLPModel` wrapper that accepts `latent_spec` and `head_spec`.

Predefined variants live in [`variants/`](https://github.com/syw-robotics/z_rl/tree/main/z_rl/models/variants):

- `EncoderMLPModel`: `ComposableModel` variant with `MLPEncoderLatentSpec` as the latent stage.
- `MoEModel`: `ComposableModel` variant whose head is replaced by a Mixture-of-Experts module. MoE head shape is controlled by `expert_hidden_dims` and `gate_hidden_dims`. This MoE implementation suppors the experts run **in parallel**.

## Export Logic

Z-RL keeps policy export ONNX-only. `MLPModel` export follows the runtime structure while using a tensor-only adapter:

```text
flat ONNX input -> latent_adapter.as_export_module() -> head -> deterministic_output
```

Export entry points:

- `model.as_onnx(...)`

Runtime latent adapters consume the structured observation `TensorDict`. If a custom adapter is not directly compatible
with the flat tensor passed by ONNX export, implement `as_export_module()` on the adapter and return a tensor-only module.

## Composition API

Preferred customization lives in [`composition/`](/home/syw/.gitrepos/z_rl/tree/main/z_rl/models/composition):

- `composition/specs.py`: base classes `LatentSpec` and `HeadSpec`
- `composition/composable_model.py`: `ComposableModel`
- `variants/`: concrete model variants and their variant-specific latent/head specs

Simple usage:

```python
from z_rl.models.composition import ComposableModel, HeadSpec, LatentSpec

model = ComposableModel(
    ...,
    latent_spec=MyLatentSpec(...),
    head_spec=MyHeadSpec(...),
)
```

`LatentSpec` defines:

- `validate(model)`
- `build_latent_adapter(model)`
- `get_latent_dim(model)`

The adapter returned by `build_latent_adapter(model)` should implement `forward(obs: TensorDict)`. It may also implement
`update_normalization(obs)` when it owns normalization statistics.

`HeadSpec` defines:

- `validate(model)`
- `build_head(model, input_dim, output_dim, activation)`

If a latent spec changes the latent width, `ComposableModel` rebuilds the head with the new dimension.

Variant-owned specs:

- `models/variants/encoder_mlp_model.py`: contains `EncoderMLPModel` and its `MLPEncoderLatentSpec`
- `models/variants/moe_model.py`: contains `MoEModel` and its `MoEHeadSpec`

When a latent or head spec only serves one concrete model variant, keep that spec in the same variant module rather
than under `composition/`.

## Maintenance Notes

- Keep runtime and export structure aligned.
- Prefer adapter modules over ad-hoc `forward()` logic.
- Keep normalization owned by the adapter that consumes the corresponding observations.
- When changing composition contracts, update this README and `z_rl/cli/plugin_templates/models.py`.
