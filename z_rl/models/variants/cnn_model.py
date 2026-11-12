# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from z_rl.modules import CNN, EmpiricalNormalization, MLP
from z_rl.models.composition import ComposableModel, GroupObsLatentAdapter, LatentSpec
from z_rl.utils import ObsSelector, resolve_nn_activation, resolve_obs_temporal_selector


class CNNObsLatentAdapter(GroupObsLatentAdapter):
    """Training path is ``GroupObsLatentAdapter``; ONNX packing lives on this subclass."""

    def __init__(
        self,
        image_obs_group: str,
        image_shape: tuple[int, ...],
        obs_groups: list[str],
        obs_group_dims: Sequence[int],
        encoders: Mapping[str, nn.Module],
        obs_normalizers: Mapping[str, nn.Module],
        append_obs: ObsSelector | None = None,
        append_obs_group: str = "policy",
    ) -> None:
        super().__init__(
            obs_groups=obs_groups,
            obs_group_dims=obs_group_dims,
            encoders=encoders,
            obs_normalizers=obs_normalizers,
            append_obs=append_obs,
            append_obs_group=append_obs_group,
        )
        self.image_obs_group = image_obs_group
        self.image_shape = image_shape
        self.vector_groups = [group for group in self.obs_groups if group != image_obs_group]
        self.vector_dims = tuple(
            int(dim) for group, dim in zip(self.obs_groups, self.obs_group_dims) if group != image_obs_group
        )

    @property
    def cnn(self) -> nn.Module:
        """CNN stem of the 2D-group encoder."""
        encoder = self.encoders[self.image_obs_group]
        return encoder[0] if isinstance(encoder, nn.Sequential) else encoder

    def as_export_module(self) -> nn.Module:
        """Return a tensor-only packing module; training ``forward`` stays TensorDict-based."""
        return _CNNObsImageExport(self)

    def export_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return dummy ONNX inputs: concat vector plus the image tensor."""
        return (torch.zeros(1, sum(self.vector_dims)), torch.zeros(1, *self.image_shape))

    def export_input_names(self) -> list[str]:
        """Return ONNX input names for the packed vector tensor and the image group."""
        return ["obs", self.image_obs_group]


@dataclass
class CNNLatentSpec(LatentSpec):
    """Encode an explicitly named 2D observation group with a CNN, then concat remaining 1D groups.

    Set ``image_obs_group`` to the image group. Remaining groups must be 1D. Training uses
    :class:`~z_rl.models.composition.GroupObsLatentAdapter`: each group is normalized and encoded on
    its own, then concatenated. ``cnn_cfg`` is forwarded to :class:`~z_rl.modules.cnn.CNN`. Optional
    ``cnn_projection_cfg`` maps flattened CNN features to a fixed width. ``append_last_obs`` appends the
    last frame of ``append_obs_group`` (default ``policy``) after the encoded groups, matching
    ``MLPEncoderLatentSpec``.
    """

    image_obs_group: str = ""
    cnn_cfg: dict[str, Any] = field(default_factory=dict)
    cnn_projection_cfg: dict[str, Any] | None = None
    append_last_obs: bool = False
    append_obs_group: str = "policy"
    _cnn_latent_dim: int = field(init=False, repr=False, default=0)

    def validate(self, model: nn.Module) -> None:
        """Require an explicit 2D ``image_obs_group`` and at least one remaining 1D group."""
        if self.cnn_cfg and all(isinstance(value, dict) for value in self.cnn_cfg.values()):
            raise ValueError(
                "`CNNLatentSpec` accepts a single `cnn_cfg` dict; encoding multiple 2D observation groups "
                "is not supported."
            )
        if self.cnn_projection_cfg and all(isinstance(value, dict) for value in self.cnn_projection_cfg.values()):
            raise ValueError("`CNNLatentSpec` accepts a single `cnn_projection_cfg` dict.")
        if self.image_obs_group not in model.obs_groups:
            raise ValueError(
                f"`CNNLatentSpec.image_obs_group` must be one of {list(model.obs_groups)}, "
                f"got '{self.image_obs_group}'."
            )
        if len(model.obs_group_shapes[self.image_obs_group]) != 3:
            raise ValueError(
                f"`CNNLatentSpec.image_obs_group` '{self.image_obs_group}' must be 2D, "
                f"got shape {model.obs_group_shapes[self.image_obs_group]}."
            )
        vector_groups = [name for name in model.obs_groups if name != self.image_obs_group]
        if not vector_groups:
            raise ValueError(
                "`CNNLatentSpec` requires at least one 1D observation group besides "
                f"`image_obs_group` '{self.image_obs_group}'."
            )
        for name in vector_groups:
            if len(model.obs_group_shapes[name]) != 1:
                raise ValueError(
                    f"Non-CNN observation groups must be 1D, got shape {model.obs_group_shapes[name]} for '{name}'."
                )
        if self.append_last_obs:
            if self.append_obs_group not in model.obs_groups:
                raise ValueError(
                    f"`CNNLatentSpec.append_last_obs` requires a '{self.append_obs_group}' observation group."
                )
            if self.image_obs_group == self.append_obs_group:
                raise ValueError(
                    "`CNNLatentSpec.append_last_obs` cannot use the 2D image group "
                    f"as '{self.append_obs_group}'."
                )

    def build(self, model: nn.Module) -> nn.Module:
        """Build a group adapter with a CNN encoder on the specified 2D observation group."""
        image_encoder = self._make_cnn_encoder(model)
        encoders = {}
        normalizers = {}
        group_dims = []

        if model.obs_normalization is False:
            normalization_cfg = None
        else:
            normalization_cfg = {} if model.obs_normalization is True else model.obs_normalization

        for name in model.obs_groups:
            shape = model.obs_group_shapes[name]
            # do encode image obs with cnn encoder
            if name == self.image_obs_group:
                encoders[name] = image_encoder
                normalizers[name] = nn.Identity()
                group_dims.append(int(shape[0] * shape[1] * shape[2]))
                continue
            # do not encode the 1D observation group
            dim = int(shape[-1])
            encoders[name] = nn.Identity()

            # build obs normalizers
            if normalization_cfg is None:
                normalizers[name] = nn.Identity()
            else:
                normalizers[name] = EmpiricalNormalization(dim, **normalization_cfg)
            group_dims.append(dim)
        append_obs = None
        if self.append_last_obs:
            append_obs = resolve_obs_temporal_selector(
                self.append_obs_group, "last", model.obs_group_time_slice_map
            )
        return CNNObsLatentAdapter(
            image_obs_group=self.image_obs_group,
            image_shape=tuple(int(v) for v in model.obs_group_shapes[self.image_obs_group]),
            obs_groups=list(model.obs_groups),
            obs_group_dims=group_dims,
            encoders=encoders,
            obs_normalizers=normalizers,
            append_obs=append_obs,
            append_obs_group=self.append_obs_group,
        )

    def get_latent_dim(self, model: nn.Module) -> int:
        """Return vector width plus the CNN width recorded while building the encoder."""
        dim = int(model.obs_dim) + self._cnn_latent_dim
        if not self.append_last_obs:
            return dim
        return dim + resolve_obs_temporal_selector(
            self.append_obs_group, "last", model.obs_group_time_slice_map
        ).dim

    def _make_cnn_encoder(self, model: nn.Module) -> nn.Sequential:
        """Build ``CNN -> optional projector`` and record the CNN latent width."""
        channels, height, width = model.obs_group_shapes[self.image_obs_group]
        cnn = CNN(input_dim=(int(height), int(width)), input_channels=int(channels), **self.cnn_cfg)
        if cnn.output_channels is not None:
            raise ValueError("The output of the CNN must be flattened before passing it to the MLP.")
        projector, self._cnn_latent_dim = _make_projector(int(cnn.output_dim), self.cnn_projection_cfg)
        return nn.Sequential(cnn, projector)


class _CNNObsImageExport(nn.Module):
    """ONNX packing: concat 1D groups as ``obs``, keep the image at native rank, then reuse group modules."""

    def __init__(self, adapter: CNNObsLatentAdapter) -> None:
        super().__init__()
        self.obs_groups = list(adapter.obs_groups)
        self.image_obs_group = adapter.image_obs_group
        self.vector_groups = list(adapter.vector_groups)
        self.vector_dims = adapter.vector_dims
        self.encoders = adapter.encoders
        self.normalizers = adapter.obs_normalizers
        self.append_obs = adapter.append_obs
        self.append_obs_group = adapter.append_obs_group

    def forward(self, obs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Split packed ``obs`` back into 1D groups, then encode with the image."""
        tensors = dict(zip(self.vector_groups, obs.split(self.vector_dims, dim=-1)))
        tensors[self.image_obs_group] = image
        encoded = []
        append_group_obs = None
        for group in self.obs_groups:
            x = self.normalizers[group](tensors[group])
            encoded.append(self.encoders[group](x))
            if group == self.append_obs_group:
                append_group_obs = x
        latent = torch.cat(encoded, dim=-1)
        if self.append_obs is None:
            return latent
        return torch.cat([latent, self.append_obs.select(append_group_obs)], dim=-1)


class CNNModel(ComposableModel):
    """Named preset that installs ``CNNLatentSpec``."""

    latent_spec_class = CNNLatentSpec

    def init_cnn_weights(self) -> None:
        """Initialize the CNN encoder with Kaiming initialization."""
        self.latent_adapter.cnn.init_cnn_weights()


def _make_projector(input_dim: int, cfg: dict[str, Any] | None) -> tuple[nn.Module, int]:
    """Build the optional post-CNN projection and return it with its output width."""
    if cfg is None:
        return nn.Identity(), input_dim
    projector_cfg = dict(cfg)
    output_dim = int(projector_cfg.pop("output_dim"))
    hidden_dims = projector_cfg.pop("hidden_dims", [])
    activation = projector_cfg.pop("activation", "elu")
    last_activation = projector_cfg.pop("last_activation", None)
    if projector_cfg:
        raise ValueError(f"Unsupported CNN projection configuration keys: {list(projector_cfg.keys())}")
    if hidden_dims:
        return MLP(input_dim, output_dim, hidden_dims, activation, last_activation), output_dim
    layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
    if last_activation is not None:
        layers.append(resolve_nn_activation(last_activation))
    return nn.Sequential(*layers), output_dim
