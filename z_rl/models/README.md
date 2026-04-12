# Models

This directory contains the neural-network model implementations used by Z-RL and the mixin system used to customize
their latent pipeline and output head.

## Overview

The core model classes are:

- `MLPModel`: base model for vector observations.
- `RNNModel`: recurrent model built on top of the MLP pipeline.
- `CNNModel`: model for mixed 1D/2D observations.
- `MoEModel`: `MLPModel` variant whose head is replaced by a Mixture-of-Experts module.

The mixin classes live in `mixins/`:

- `BaseModelMixin`: shared hook interface for latent adapters and custom heads.
- `MLPEncoderMixin`: example latent mixin that replaces the latent adapter and changes latent dimensionality.
- `MoEHeadMixin`: head mixin that replaces the default head with an MoE head.

Two composed helper models are exported in `__init__.py`:

- `MLPEncoderMLPModel`
- `MLPEncoderMoEModel`

## Current Structure

The model pipeline is organized around three stages:

1. Observation preprocessing
2. Latent transformation
3. Output head

For `MLPModel`, the flow is:

```text
obs groups -> concat obs -> obs_normalizer -> latent_adapter -> head -> distribution -> output
```

For `RNNModel`, the flow is:

```text
obs groups -> concat obs -> obs_normalizer -> rnn -> head -> distribution -> output
```

For `CNNModel`, the flow is:

```text
1D obs -> obs_normalizer
2D obs -> cnn / projector
concat -> head -> distribution -> output
```

## Export Logic

`MLPModel` export is designed to follow the same structure as runtime inference:

```text
normalized input -> latent_adapter -> head -> deterministic_output
```

This is why latent customization is now centered on `latent_adapter`: it makes JIT and ONNX export reuse the same
structure instead of requiring per-mixin export wrappers.

Current export entry points:

- `model.as_jit()`
- `model.as_onnx(...)`

Important design note:

- `head` customization is naturally export-friendly because the export wrappers copy `model.head`.
- `MLPModel` latent customization is export-friendly because the export wrappers copy `model.latent_adapter`.
- `RNNModel` and `CNNModel` currently keep their own export wrappers and do not share the `latent_adapter` path.

## Mixin Design

`BaseModelMixin` exposes two extension surfaces:

### Latent customization

- `build_latent_adapter()`
- `install_latent_adapter()`
- `get_latent_dim()`

Use this path when you want to change the latent before it enters the final head.

Contract:

- The adapter must be an exportable `nn.Module`.
- The adapter currently receives the normalized latent tensor as its single input.
- If the adapter changes latent dimensionality, the mixin must return the new size from `get_latent_dim()`.

### Head customization

- `build_custom_head(input_dim, output_dim, activation)`
- `install_custom_head(...)`

Use this path when you want to replace the final head module but keep the upstream latent pipeline.

## How To Use a Mixin

Typical composition order is:

```python
class MyActorModel(MyHeadMixin, MyLatentMixin, MLPModel):
    pass
```

General rule:

- latent-related mixins should appear before the concrete model if they need to affect head initialization
- head-related mixins should appear before the concrete model if they replace `self.head`

For models that need explicit post-init installation, do it in the concrete model `__init__`.

## How To Implement a Latent Mixin

If your mixin changes the latent, implement:

1. `build_latent_adapter()`
2. `get_latent_dim()`

Minimal pattern:

```python
class MyLatentAdapter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MyLatentMixin(BaseModelMixin):
    modifies_latent = True

    def build_latent_adapter(self) -> nn.Module:
        return MyLatentAdapter()

    def get_latent_dim(self) -> int:
        return ...
```

If the latent size changes, the new dimension must be available before the parent model initializes `self.head`.
`MLPEncoderMixin` is the reference implementation for that pattern.

## How To Implement a Head Mixin

Implement:

1. `build_custom_head(...)`
2. optional `validate_mixin_contract()`

Minimal pattern:

```python
class MyHeadMixin(BaseModelMixin):
    modifies_head = True

    def build_custom_head(self, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        return nn.Linear(input_dim, output_dim)
```

If the model is stochastic, the head output dimension should match `distribution.input_dim`, not necessarily the final
action dimension. `MoEHeadMixin` shows the current reference pattern.

## Reference Implementations

Use these files as the canonical examples:

- `mixins/latent_mixin.py`: latent adapter mixin that changes latent dimensionality
- `mixins/head_mixin.py`: head replacement mixin
- `moe_model.py`: concrete model that applies a head mixin

## Maintenance Notes

- Keep runtime and export structure aligned whenever possible.
- Prefer adapter modules over ad-hoc logic inside model `forward()`.
- When changing mixin contracts, update both this README and `z_rl/cli/plugin_templates/models.py`.
