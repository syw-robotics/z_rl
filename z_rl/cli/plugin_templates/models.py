"""Model template sources for plugin scaffold generation."""

MODELS_INIT_TEMPLATE = (
    "from .my_model import MyActorModel, MyCriticModel\n\n"
    "__all__ = [\"MyActorModel\", \"MyCriticModel\"]\n"
)

MODELS_MODEL_TEMPLATE = """from __future__ import annotations

import torch
from tensordict import TensorDict

from z_rl.models.mixins.model_mixin import BaseModelMixin
from z_rl.models.mlp_model import MLPModel
from z_rl.modules import HiddenState


class MyModelMixin(BaseModelMixin):
    \"\"\"Example generic model mixin.

    This shows how to customize latent construction without depending on encoder mixins.
    \"\"\"

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        latent = MLPModel.get_latent(self, obs, masks, hidden_state)
        # Example hook: you can transform/append features here.
        return latent


class MyActorModel(MyModelMixin, MLPModel):
    \"\"\"Example actor model with a generic custom mixin.\"\"\"
    pass


class MyCriticModel(MyModelMixin, MLPModel):
    \"\"\"Example critic model with a generic custom mixin.\"\"\"
    pass
"""
