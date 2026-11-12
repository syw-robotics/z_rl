# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import inspect
from typing import Any

import torch

from z_rl.utils import bind_matching_fields, resolve_spec

from ..mlp_model import MLPModel
from .specs import HeadSpec, LatentSpec

SpecRef = LatentSpec | HeadSpec | dict[str, Any] | str | type | None

_RESERVED_MODEL_KWARGS = frozenset(inspect.signature(MLPModel.__init__).parameters) | {"latent_spec", "head_spec"}


class ComposableModel(MLPModel):
    """MLPModel variant that composes optional latent and head specs during base initialization.

    Named subclasses can set ``latent_spec_class`` / ``head_spec_class`` so ``class_name`` still
    works without nested spec dicts. Leftover constructor kwargs that match spec init fields are
    bound onto the specs; keys accepted by ``MLPModel`` stay on the model (for example ``activation``).
    """

    latent_spec_class: SpecRef = None
    head_spec_class: SpecRef = None

    def __init__(self, *args, latent_spec: SpecRef = None, head_spec: SpecRef = None, **kwargs) -> None:
        """Resolve specs, bind leftover spec fields, then initialize the base model."""
        # resolve the latent spec and head spec
        if latent_spec is None:
            latent_spec = type(self).latent_spec_class
        if head_spec is None:
            head_spec = type(self).head_spec_class
        self.latent_spec = resolve_spec(latent_spec)
        self.head_spec = resolve_spec(head_spec)
        # bind the leftover spec fields
        if self.latent_spec is not None:
            bind_matching_fields(self.latent_spec, kwargs, exclude=_RESERVED_MODEL_KWARGS)
        if self.head_spec is not None:
            bind_matching_fields(self.head_spec, kwargs, exclude=_RESERVED_MODEL_KWARGS)
        # initialize the base model
        super().__init__(*args, **kwargs)

    def build_latent_adapter(self) -> torch.nn.Module:
        """Build the configured latent adapter or fall back to the base identity adapter."""
        if self.latent_spec is None:
            return super().build_latent_adapter()
        # validate user specified safety checks
        self.latent_spec.validate(self)
        adapter = self.latent_spec.build(self)
        self.latent_dim = self.latent_spec.get_latent_dim(self)
        return adapter

    def build_head(
        self,
        input_dim: int,
        output_dim: int | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> torch.nn.Module:
        """Build the configured head or fall back to the default MLP head."""
        if self.head_spec is None:
            return super().build_head(input_dim, output_dim, hidden_dims, activation)
        # validate user specified safety checks
        self.head_spec.validate(self)
        # return the head module
        return self.head_spec.build(self, input_dim, output_dim, activation)