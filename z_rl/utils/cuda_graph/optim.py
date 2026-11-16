"""Optimizer helpers for CUDA-captured training updates."""

import torch
from torch import nn
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

from ..opt import OptimizerGroup


def clip_grad_norm(parameters, max_norm):
    parameters = list(parameters)
    on_cuda = parameters[0].is_cuda
    norm = nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=not on_cuda)
    if on_cuda:
        # Fail on the device before the optimizer; no scalar readback per minibatch.
        torch._assert_async(torch.isfinite(norm), "Non-finite gradient norm")
    return norm


class TensorLRMuon(torch.optim.Muon):
    """Native Muon update with a device LR; native step converts tensor LR to a scalar."""

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Captured Muon does not support closures")
        for group in self.param_groups:
            params, grads, buffers = [], [], []
            self._init_group(group, params, grads, buffers)
            for param, grad, buffer in zip(params, grads, buffers):
                buffer.lerp_(grad, 1 - group["momentum"])
                update = grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                update = _zeropower_via_newtonschulz(update, group["ns_coefficients"], group["ns_steps"], group["eps"])
                lr = _adjust_lr(group["lr"], group["adjust_lr_fn"], param.shape)
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(-lr * update.to(param.dtype))


def make_capturable(optimizer):
    """Call before training; preserve Muon/AdamW routing and checkpoint structure."""
    if isinstance(optimizer, OptimizerGroup):
        optimizer.optimizers = [make_capturable(opt) for opt in optimizer.optimizers]
        optimizer.param_groups = [group for opt in optimizer.optimizers for group in opt.param_groups]
        return optimizer
    if isinstance(optimizer, torch.optim.Muon):
        native = optimizer
        optimizer = TensorLRMuon(native.param_groups)
        optimizer.load_state_dict(native.state_dict())
    elif isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
        for group in optimizer.param_groups:
            group.update(capturable=True, fused=True)
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
