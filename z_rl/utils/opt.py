# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain

import torch
import torch.nn as nn

from z_rl.utils.utils import resolve_optimizer


class OptimizerGroup(torch.optim.Optimizer):
    """Wrapper around multiple optimizers with a single optimizer-like interface."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        if len(optimizers) == 0:
            raise ValueError("OptimizerGroup requires at least one optimizer.")

        all_params = []
        for opt in optimizers:
            for group in opt.param_groups:
                all_params.extend(group["params"])
        if len(all_params) == 0:
            raise ValueError("OptimizerGroup underlying optimizers have no parameters.")

        super().__init__(params=all_params, defaults={})
        self.optimizers = optimizers
        self.param_groups = [group for opt in optimizers for group in opt.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = self.optimizers[0].step(closure)
            for opt in self.optimizers[1:]:
                opt.step()
            return loss

        for opt in self.optimizers:
            step_loss = opt.step()
            if loss is None:
                loss = step_loss
        return loss

    def zero_grad(self, set_to_none: bool | None = None):
        for opt in self.optimizers:
            if set_to_none is None:
                opt.zero_grad()
            else:
                opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "class": self.__class__.__name__,
        }

    def load_state_dict(self, state_dict):
        opt_states = state_dict.get("optimizers")
        if not isinstance(opt_states, list):
            raise ValueError("OptimizerGroup checkpoint must contain an 'optimizers' list.")
        if len(opt_states) != len(self.optimizers):
            raise ValueError(
                f"OptimizerGroup checkpoint has {len(opt_states)} optimizers, "
                f"but the current group has {len(self.optimizers)}."
            )
        for opt, opt_state in zip(self.optimizers, opt_states):
            saved_sizes = [len(group["params"]) for group in opt_state["param_groups"]]
            current_sizes = [len(group["params"]) for group in opt.param_groups]
            if saved_sizes != current_sizes:
                raise ValueError("OptimizerGroup checkpoint parameter groups do not match the current optimizer.")
        for opt, opt_state in zip(self.optimizers, opt_states):
            opt.load_state_dict(opt_state)
        # Loading replaces the wrapped optimizers' parameter-group dictionaries.
        self.param_groups = [group for opt in self.optimizers for group in opt.param_groups]


def build_optimizer(
    modules: Sequence[nn.Module],
    *,
    learning_rate: float,
    optimizer: str = "adam",
    use_muon: bool = False,
    muon_adamw_weight_decay: float = 0.01,
    weight_decay: float | None = None,
) -> torch.optim.Optimizer:
    """Use the named optimizer, or split trainable parameters between Muon and AdamW.

    With ``optimizer="muon_adamw"``, Muon takes 2-D parameters unless the model marks them
    with ``_non_muon``. ``weight_decay`` applies to the named optimizer or the Muon half
    of the split. The AdamW half uses ``muon_adamw_weight_decay``. The legacy ``use_muon=True`` flag still overrides the optimizer name.
    """
    if use_muon:
        optimizer = "muon_adamw"
    params = chain.from_iterable(module.parameters() for module in modules)
    decay = {} if weight_decay is None else {"weight_decay": weight_decay}
    if optimizer.lower() != "muon_adamw":
        return resolve_optimizer(optimizer)(params, lr=learning_rate, **decay)
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError('optimizer="muon_adamw" requires a PyTorch build with torch.optim.Muon.')

    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for param in dict.fromkeys(params):
        if not param.requires_grad:
            continue
        if param.ndim == 2 and not getattr(param, "_non_muon", False):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    if not (muon_params or adamw_params):
        raise ValueError('optimizer="muon_adamw" requires at least one trainable parameter.')

    optimizers: list[torch.optim.Optimizer] = []
    if muon_params:
        optimizers.append(torch.optim.Muon(muon_params, lr=learning_rate, adjust_lr_fn="match_rms_adamw", **decay))
    if adamw_params:
        optimizers.append(
            torch.optim.AdamW(adamw_params, lr=learning_rate, weight_decay=muon_adamw_weight_decay, fused=True)
        )
    return OptimizerGroup(optimizers)
