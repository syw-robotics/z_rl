"""Muon with a device learning rate, imported only when capturing Muon."""

import torch
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz


class TensorLRMuon(torch.optim.Muon):
    """Keep the learning rate on the device instead of reading it as a Python scalar."""

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
