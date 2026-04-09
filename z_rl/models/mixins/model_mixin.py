from __future__ import annotations

import abc
import torch
from tensordict import TensorDict

from z_rl.modules import HiddenState


class BaseModelMixin(abc.ABC):
    """Template mixin for custom model extensions.

    Data flow: ``obs -> get_latent (optional transform_latent hook) -> head/distribution in base model -> output (optional transform_output hook)``.

    Usage:
        1. Inherit from this mixin before a concrete z_rl model class (for example ``MLPModel``).
        2. Override :meth:`transform_latent` and/or :meth:`transform_output` when custom behavior is needed.
        3. Keep ``super()`` calls in overridden methods to preserve base model behavior.

    Notes:
        - The default implementation is identity, so this mixin is safe as a template base class.
        - Export behavior (``as_jit``/``as_onnx``) remains owned by the concrete model.
    """

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build latent with base model logic, then apply an optional latent transform hook."""
        #  latent = super().get_latent(obs, masks, hidden_state)
        raise NotImplementedError
