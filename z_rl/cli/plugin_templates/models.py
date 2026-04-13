"""Model template sources for plugin scaffold generation."""

MODELS_INIT_TEMPLATE = (
    "from .my_model import MyActorModel, MyCriticModel\n\n"
    "__all__ = [\"MyActorModel\", \"MyCriticModel\"]\n"
)

MODELS_MODEL_TEMPLATE = """from __future__ import annotations

import torch.nn as nn

from z_rl.models.composition import ComposableModel, HeadSpec, LatentSpec


class _MyLatentAdapter(nn.Module):
    \"\"\"Example latent adapter.

    Latent adapters operate on the model latent input tensor and should stay exportable
    for runtime, JIT, and ONNX paths. This example keeps the latent dimensionality unchanged.
    \"\"\"

    def forward(self, x):
        return x


class MyLatentSpec(LatentSpec):
    \"\"\"Example latent spec.

    This shows how to replace the latent adapter without reimplementing the full
    latent pipeline or export wrappers. If your adapter changes the latent
    dimensionality, override `get_latent_dim()` so the final head can be rebuilt
    with the new input dimension.
    \"\"\"

    def validate(self, model) -> None:
        return None

    def build_latent_adapter(self, model) -> nn.Module:
        # Replace this with your own exportable latent adapter.
        return _MyLatentAdapter()

    def get_latent_dim(self, model) -> int:
        return model.obs_dim


class MyHeadSpec(HeadSpec):
    \"\"\"Example head spec.

    This shows how to replace the model head after the parent model has already
    initialized its latent pipeline and output distribution.
    \"\"\"

    def validate(self, model) -> None:
        return None

    def build_head(self, model, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        # Example head: a tiny MLP block ending at the requested output dimension.
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )


class MyActorModel(ComposableModel):
    \"\"\"Example actor model with both latent and head customizations.\"\"\"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, latent_spec=MyLatentSpec(), head_spec=MyHeadSpec(), **kwargs)


class MyCriticModel(ComposableModel):
    \"\"\"Example critic model with only latent customization.\"\"\"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, latent_spec=MyLatentSpec(), **kwargs)
"""
