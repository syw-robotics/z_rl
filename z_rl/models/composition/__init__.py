"""Explicit model composition APIs."""

from .adapters import GroupObsLatentAdapter, ObsLatentAdapter
from .composable_model import ComposableModel
from .specs import HeadSpec, LatentSpec

__all__ = [
    "ComposableModel",
    "GroupObsLatentAdapter",
    "ObsLatentAdapter",
    "LatentSpec",
    "HeadSpec",
]
