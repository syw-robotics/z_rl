"""Optimizer helpers for CUDA-captured training updates."""

import torch
from torch import nn

from ..opt import OptimizerGroup


def clip_grad_norm(parameters, max_norm):
    parameters = list(parameters)
    on_cuda = parameters[0].is_cuda
    norm = nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=not on_cuda)
    if on_cuda:
        # Fail on the device before the optimizer; no scalar readback per minibatch.
        torch._assert_async(torch.isfinite(norm), "Non-finite gradient norm")
    return norm


def make_capturable(optimizer):
    """Prepare an optimizer for capture before training or after loading a checkpoint."""
    if isinstance(optimizer, OptimizerGroup):
        optimizer.optimizers = [make_capturable(opt) for opt in optimizer.optimizers]
        optimizer.param_groups = [group for opt in optimizer.optimizers for group in opt.param_groups]
        return optimizer
    if isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
        for group in optimizer.param_groups:
            group.update(capturable=True, fused=True, foreach=False)
        # Eager Adam checkpoints can have CPU step counters.
        for param, state in optimizer.state.items():
            if "step" in state:
                state["step"] = state["step"].to(device=param.device, dtype=torch.float32)
    elif isinstance(optimizer, getattr(torch.optim, "Muon", ())):
        from .muon import TensorLRMuon

        if not isinstance(optimizer, TensorLRMuon):
            native = optimizer
            optimizer = TensorLRMuon(native.param_groups)
            optimizer.load_state_dict(native.state_dict())
    else:
        raise TypeError(f"Unsupported captured optimizer: {type(optimizer).__name__}")
    for group in optimizer.param_groups:
        group["lr"] = torch.as_tensor(group["lr"], device=group["params"][0].device).clone()
    return optimizer


def set_learning_rate(group, value):
    if isinstance(group["lr"], torch.Tensor):
        group["lr"].fill_(value)
    else:
        group["lr"] = value
