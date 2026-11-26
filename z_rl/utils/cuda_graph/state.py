"""Track tensor storage and Python configuration used by captured updates."""

import torch
from torch import nn

from ..opt import OptimizerGroup


def graph_state(modules, optimizers):
    """Tensors whose storage is referenced by a captured update."""
    tensors = [tensor for module in modules for tensor in (*module.parameters(), *module.buffers())]
    for optimizer in optimizers:
        parts = optimizer.optimizers if isinstance(optimizer, OptimizerGroup) else [optimizer]
        for part in parts:
            tensors.extend(value for state in part.state.values() for value in state.values()
                           if isinstance(value, torch.Tensor))
            tensors.extend(group["lr"] for group in part.param_groups if isinstance(group["lr"], torch.Tensor))
    return tensors


def _config_snapshot(value):
    """Copy Python configuration by value, without reading any device tensors."""
    if isinstance(value, dict):
        return tuple((key, _config_snapshot(item)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_config_snapshot(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("Capture config must contain Python scalars and containers; track tensors with state")


def graph_config(objects, optimizers):
    """Collect module modes, explicit capture config, and optimizer Python hyperparameters."""
    config = []
    for obj in objects:
        modules = obj.modules() if isinstance(obj, nn.Module) else (obj,)
        for module in modules:
            if isinstance(module, nn.Module):
                config.append(module.training)
            hook = getattr(module, "capture_config", None)
            if hook is not None:
                config.append(hook())
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            config.append({key: value for key, value in group.items()
                           if key != "params" and not isinstance(value, torch.Tensor)})
    return config
