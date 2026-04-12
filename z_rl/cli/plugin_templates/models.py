"""Model template sources for plugin scaffold generation."""

MODELS_INIT_TEMPLATE = (
    "from .my_model import MyActorModel, MyCriticModel\n\n"
    "__all__ = [\"MyActorModel\", \"MyCriticModel\"]\n"
)

MODELS_MODEL_TEMPLATE = """from __future__ import annotations

import torch.nn as nn

from z_rl.models.mixins.base_mixin import BaseModelMixin
from z_rl.models.mlp_model import MLPModel


class MyLatentAdapter(nn.Module):
    \"\"\"Example latent adapter.

    Latent adapters receive both the raw concatenated observations and the normalized
    latent, which makes them compatible with runtime, JIT, and ONNX export paths.
    This example keeps the latent dimensionality unchanged.
    \"\"\"

    def forward(self, raw_x, normalized_x):
        return normalized_x


class MyLatentMixin(BaseModelMixin):
    \"\"\"Example latent mixin.

    This shows how to replace the latent adapter without reimplementing the full
    latent pipeline or export wrappers. If your adapter changes the latent
    dimensionality, you should also override `get_latent_dim()` so the
    final head is initialized with the new input dimension.
    \"\"\"

    modifies_latent = True

    def build_latent_adapter(self) -> nn.Module:
        # Example hook: replace this with your own exportable latent adapter.
        return MyLatentAdapter()


class MyHeadMixin(BaseModelMixin):
    \"\"\"Example head mixin.

    This shows how to replace the model head after the parent model has already
    initialized its latent pipeline and output distribution.
    \"\"\"

    modifies_head = True

    def build_custom_head(self, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        # Example head: a tiny MLP block ending at the requested output dimension.
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )

    def validate_mixin_contract(self) -> None:
        # Put mixin-specific checks here when your custom head depends on extra attrs.
        return None


class MyActorModel(MyHeadMixin, MyLatentMixin, MLPModel):
    \"\"\"Example actor model with both latent and head customizations.\"\"\"

    def __init__(self, *args, output_dim: int, activation: str = "elu", **kwargs) -> None:
        super().__init__(*args, output_dim=output_dim, activation=activation, **kwargs)
        self.validate_mixin_contract()
        head_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.install_custom_head(self.get_latent_dim(), head_output_dim, activation)


class MyCriticModel(MyLatentMixin, MLPModel):
    \"\"\"Example critic model with only latent customization.\"\"\"
    pass
"""
