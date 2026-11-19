"""CUDA capture helpers for training updates with static tensor shapes."""

from .optim import clip_grad_norm, make_capturable, set_learning_rate
from .state import graph_config, graph_state
from .update import GraphedUpdate


def __getattr__(name):
    if name == "TensorLRMuon":
        from .muon import TensorLRMuon

        return TensorLRMuon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GraphedUpdate",
    "clip_grad_norm",
    "graph_config",
    "graph_state",
    "make_capturable",
    "set_learning_rate",
]
