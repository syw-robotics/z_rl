# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pack and read per-child model entries inside a training checkpoint."""

from __future__ import annotations

import torch
import torch.nn as nn


def model_checkpoint_entries(role: str, model: nn.Module) -> dict[str, object]:
    """Flat state dict plus per-child module state dicts for one checkpoint role."""
    entries: dict[str, object] = {f"{role}_state_dict": model.state_dict()}
    export_modules = getattr(model, "module_state_dicts", None)
    if callable(export_modules):
        entries[f"{role}_modules"] = export_modules()
    return entries


def module_state_dict_from_checkpoint(
    checkpoint: dict, module_name: str, *, role: str
) -> dict[str, torch.Tensor]:
    """Read one child module from ``{role}_modules`` or an older flat state dict.

    A ``teacher`` role with no teacher entries falls back to ``actor``, matching distillation loads
    from an RL checkpoint.
    """
    roles = [role]
    if role == "teacher" and not _checkpoint_role_present(checkpoint, "teacher"):
        roles.append("actor")

    for candidate in roles:
        modules = checkpoint.get(f"{candidate}_modules")
        if isinstance(modules, dict) and module_name in modules:
            module_state = modules[module_name]
            if not isinstance(module_state, dict):
                raise TypeError(f"`{candidate}_modules[{module_name}]` is not a state dict.")
            return module_state
        flat = checkpoint.get(f"{candidate}_state_dict")
        prefix = f"{module_name}."
        if isinstance(flat, dict) and any(key.startswith(prefix) for key in flat):
            return {key[len(prefix) :]: value for key, value in flat.items() if key.startswith(prefix)}
        if _checkpoint_role_present(checkpoint, candidate):
            available = _checkpoint_module_names(checkpoint, candidate)
            raise KeyError(
                f"Module `{module_name}` not found in `{candidate}` checkpoint entries. Available: {available}."
            )
    expected = f"`{role}_modules` or `{role}_state_dict`"
    if role == "teacher":
        expected += " (or `actor_*` when loading a teacher from RL)"
    raise KeyError(f"No `{role}` module checkpoint found. Expected {expected}.")


def _checkpoint_role_present(checkpoint: dict, role: str) -> bool:
    return f"{role}_modules" in checkpoint or f"{role}_state_dict" in checkpoint


def _checkpoint_module_names(checkpoint: dict, role: str) -> list[str]:
    modules = checkpoint.get(f"{role}_modules")
    if isinstance(modules, dict):
        return sorted(modules)
    flat = checkpoint.get(f"{role}_state_dict")
    if isinstance(flat, dict):
        return sorted({key.split(".", 1)[0] for key in flat})
    return []
