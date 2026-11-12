from __future__ import annotations

import abc

import torch.nn as nn


class LatentSpec(abc.ABC):
    """Abstract base class for replacing the latent adapter of an ``MLPModel``-compatible model.

    Runtime latent adapters consume the structured observation ``TensorDict``. If an adapter needs different behavior
    for ONNX export, it should expose ``as_export_module()`` returning a tensor-only module.

    Required overrides: ``build`` and ``get_latent_dim``. ``validate`` is a no-op by default.
    Omit ``latent_spec`` to keep the identity obs-width path (``model.obs_dim``).
    """

    def validate(self, model: nn.Module) -> None:
        """Validate spec-specific assumptions against the initialized model."""
        del model

    @abc.abstractmethod
    def build(self, model: nn.Module) -> nn.Module:
        """Build the latent adapter installed on the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_latent_dim(self, model: nn.Module) -> int:
        """Return the latent dimensionality consumed by the model head.

        Invoked after ``build``, so implementations may read widths from constructed modules.
        """
        raise NotImplementedError


class HeadSpec(abc.ABC):
    """Abstract base class for replacing the output head of an ``MLPModel``-compatible model.

    Required override: ``build``. ``validate`` is a no-op by default.
    """

    def validate(self, model: nn.Module) -> None:
        """Validate spec-specific assumptions against the initialized model."""
        del model

    @abc.abstractmethod
    def build(self, model: nn.Module, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        """Build the output head installed on the model."""
        raise NotImplementedError
