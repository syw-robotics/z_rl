"""Explicit model composition APIs."""

from .composable_model import ComposableModel
from .specs import HeadSpec, LatentSpec

__all__ = [
    "ComposableModel",
    "LatentSpec",
    "HeadSpec",
]
