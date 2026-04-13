from __future__ import annotations

import abc

import torch.nn as nn


class LatentSpec(abc.ABC):
    """Abstract base class for replacing the latent adapter of an ``MLPModel``-compatible model."""

    @abc.abstractmethod
    def validate(self, model: nn.Module) -> None:
        """Validate spec-specific assumptions against the initialized model."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_latent_adapter(self, model: nn.Module) -> nn.Module:
        """Build the latent adapter installed on the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_latent_dim(self, model: nn.Module) -> int:
        """Return the latent dimensionality produced by the installed adapter."""
        raise NotImplementedError


class HeadSpec(abc.ABC):
    """Abstract base class for replacing the output head of an ``MLPModel``-compatible model."""

    @abc.abstractmethod
    def validate(self, model: nn.Module) -> None:
        """Validate spec-specific assumptions against the initialized model."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_head(self, model: nn.Module, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        """Build the output head installed on the model."""
        raise NotImplementedError

