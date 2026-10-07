# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
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
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

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
        opt_states = state_dict.get("optimizers", None)
        if opt_states is None:
            return
        if len(opt_states) != len(self.optimizers):
            warnings.warn(
                f"OptimizerGroup state has {len(opt_states)} optimizers, "
                f"but current instance has {len(self.optimizers)}. "
                "Loading states for the matching prefix only."
            )
        for opt, opt_state in zip(self.optimizers, opt_states):
            opt.load_state_dict(opt_state)


class MuonAdamWWrapper(OptimizerGroup):
    """Split parameters between Muon (hidden 2-D weights) and AdamW (everything else).

    Muon is not a drop-in replacement for Adam/AdamW. In typical usage, only hidden-layer
    ``Linear.weight`` tensors (2-D, square-ish matrices) are optimized with Muon; biases,
    normalization parameters, policy std heads, and input/output linear layers stay on AdamW.

    Parameter routing:

    - 2-D weights without ``_non_muon`` -> ``torch.optim.Muon``
    - all other parameters (1-D biases/std, marked 2-D heads, etc.) -> ``torch.optim.AdamW``

    Models can mark specific weights with ``param._non_muon = True`` (see :class:`z_rl.modules.MLP`)
    to keep them on AdamW even when they are 2-D.

    Requires a PyTorch build that exposes ``torch.optim.Muon`` (approximately 2.9+).
    """

    def __init__(
        self,
        modules: list[torch.nn.Module],
        lr: float,
        weight_decay: float = 0.01,
        ignore_frozen: bool = True,
    ):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "use_muon=True requires a PyTorch build with torch.optim.Muon. "
                "Upgrade PyTorch or set use_muon=False."
            )

        seen: set[int] = set()
        muon_params: list[torch.nn.Parameter] = []
        adamw_params: list[torch.nn.Parameter] = []
        for module in modules:
            for _, param in module.named_parameters():
                if id(param) in seen:
                    continue
                if ignore_frozen and not param.requires_grad:
                    continue
                seen.add(id(param))
                # Muon only applies to hidden 2-D weights; everything else uses AdamW.
                if param.dim() == 2 and not getattr(param, "_non_muon", False):
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

        optimizers: list[torch.optim.Optimizer] = []
        if len(muon_params) > 0:
            muon = torch.optim.Muon(muon_params, lr=lr, adjust_lr_fn="match_rms_adamw")
            optimizers.append(muon)
        if len(adamw_params) > 0:
            try:
                adamw = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay, fused=True)
            except TypeError:
                adamw = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)
            optimizers.append(adamw)
        if len(optimizers) == 0:
            raise ValueError(
                "MuonAdamWWrapper: no parameters were assigned to Muon or AdamW. "
                "With ignore_frozen=True, every parameter may have requires_grad=False; "
                "unfreeze at least some weights or pass ignore_frozen=False."
            )
        super().__init__(optimizers)


def build_optimizer(
    modules: Sequence[nn.Module],
    *,
    learning_rate: float,
    optimizer: str = "adam",
    use_muon: bool = False,
    muon_adamw_weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Build the optimizer shared by PPO and distillation.

    When ``use_muon`` is enabled, hidden 2-D weights go to Muon and the remaining parameters stay on AdamW.
    The named ``optimizer`` is ignored in that case. Otherwise every parameter of ``modules`` uses that optimizer.
    """
    if use_muon:
        return MuonAdamWWrapper(list(modules), lr=learning_rate, weight_decay=muon_adamw_weight_decay)
    params = chain.from_iterable(module.parameters() for module in modules)
    return resolve_optimizer(optimizer)(params, lr=learning_rate)
