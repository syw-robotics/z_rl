"""Model template sources for plugin scaffold generation."""

MODELS_INIT_TEMPLATE = (
    "from .my_model import MyHeadSpec, MyLatentSpec\n\n"
    "__all__ = [\"MyLatentSpec\", \"MyHeadSpec\"]\n"
)

MODELS_MODEL_TEMPLATE = """from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from z_rl.modules import EmpiricalNormalization, MLP
from z_rl.models.composition import HeadSpec, LatentSpec, ObsLatentAdapter


@dataclass
class MyLatentSpec(LatentSpec):
    \"\"\"Example latent spec. `build` and `get_latent_dim` are required.

    Use `ObsLatentAdapter` for concat-then-encode. Use `GroupObsLatentAdapter` when each
    observation group needs its own normalizer and encoder. Custom adapters should implement
    `as_export_module()`; runtime `forward` receives a TensorDict, ONNX receives a concat tensor.
    Adapters with non-vector ONNX inputs should also implement `export_dummy_inputs()` and
    `export_input_names()`.
    \"\"\"

    latent_dim: int = 128

    def build(self, model) -> nn.Module:
        if model.obs_normalization is False:
            obs_normalizer = nn.Identity()
        else:
            normalization_cfg = {} if model.obs_normalization is True else model.obs_normalization
            obs_normalizer = EmpiricalNormalization(model.obs_dim, **normalization_cfg)
        return ObsLatentAdapter(
            obs_groups=model.obs_groups,
            obs_normalizer=obs_normalizer,
            encoder=MLP(model.obs_dim, self.latent_dim, [256], "elu"),
        )

    def get_latent_dim(self, model) -> int:
        del model
        return self.latent_dim


class MyHeadSpec(HeadSpec):
    \"\"\"Example head spec. Only `build` is required.\"\"\"

    def build(self, model, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        del model, activation
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )
"""
