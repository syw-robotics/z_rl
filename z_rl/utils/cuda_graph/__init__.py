"""CUDA capture helpers for model-based updates, retaining native optimizer state.

Import the public helpers directly from ``z_rl.utils.cuda_graph``.

Modules:
    update: GraphedUpdate warm-up, capture, replay and invalidation.
    state: Tensor state tracking and Python configuration snapshots.
    optim: Capturable optimizers, learning rates and gradient clipping.
"""

from .optim import TensorLRMuon, clip_grad_norm, make_capturable, set_learning_rate
from .state import graph_config, graph_state
from .update import GraphedUpdate

__all__ = [
    "GraphedUpdate",
    "TensorLRMuon",
    "clip_grad_norm",
    "graph_config",
    "graph_state",
    "make_capturable",
    "set_learning_rate",
]
