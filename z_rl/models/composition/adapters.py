# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict

from z_rl.utils import ObsSelector


class ObsLatentAdapter(nn.Module):
    """Concatenate observation groups, normalize once, then encode.

    Data flow: ``concat(obs_groups) -> normalizer -> encoder -> optional last-frame policy obs``.

    Runtime ``forward`` always receives a ``TensorDict``. ONNX export uses ``as_export_module()``,
    which receives the already-concatenated tensor.

    ``append_obs`` is typically ``resolve_obs_temporal_selector(append_obs_group, "last", ...)``.
    It is applied to the concatenated, normalized observation, then concatenated onto the encoded
    latent. Requires ``append_obs_group`` to be one of ``obs_groups`` when set.
    """

    def __init__(
        self,
        obs_groups: list[str],
        obs_normalizer: nn.Module,
        encoder: nn.Module | None = None,
        append_obs: ObsSelector | None = None,
        append_obs_group: str = "policy",
    ) -> None:
        super().__init__()
        self.obs_groups = list(obs_groups)
        self.append_obs_group = append_obs_group
        if append_obs is not None and append_obs_group not in self.obs_groups:
            raise ValueError(f"`append_obs` requires a '{append_obs_group}' observation group.")
        self.obs_normalizer = obs_normalizer
        self._obs_normalizer_update = getattr(obs_normalizer, "update", None)
        self.encoder = nn.Identity() if encoder is None else encoder
        self.append_obs = append_obs

    def forward(self, obs: TensorDict) -> torch.Tensor:
        # Avoid torch.cat overhead when only one observation group is present.
        if len(self.obs_groups) == 1:
            x = obs[self.obs_groups[0]]
        else:
            x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        return self.encode_concat(x)

    def encode_concat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        latent = self.encoder(x)
        if self.append_obs is None:
            return latent
        return torch.cat([latent, self.append_obs.select(x)], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self._obs_normalizer_update is None:
            return
        # Avoid torch.cat overhead when only one observation group is present.
        if len(self.obs_groups) == 1:
            x = obs[self.obs_groups[0]]
        else:
            x = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        self._obs_normalizer_update(x)

    def as_export_module(self) -> nn.Module:
        return _ObsLatentAdapterExport(self)


class GroupObsLatentAdapter(nn.Module):
    """Normalize and encode each observation group, then concatenate.

    Data flow: ``group_i -> normalizer_i -> encoder_i``, then
    ``concat(encoded_i) -> optional last-frame policy obs``.

    Use this for heterogeneous streams such as proprioception and exteroception.
    Runtime ``forward`` always receives a ``TensorDict``. ONNX export uses ``as_export_module()``,
    which splits the concatenated tensor back into groups.

    ``append_obs`` matches ``ObsLatentAdapter``: typically
    ``resolve_obs_temporal_selector(append_obs_group, "last", ...)``. It is applied to the
    normalized ``append_obs_group``, then concatenated onto the encoded latent. Requires that
    group to be one of ``obs_groups`` when set. ``append_obs_group`` defaults to ``policy``.
    """

    def __init__(
        self,
        obs_groups: list[str],
        obs_group_dims: Sequence[int],
        encoders: Mapping[str, nn.Module],
        obs_normalizers: Mapping[str, nn.Module],
        append_obs: ObsSelector | None = None,
        append_obs_group: str = "policy",
    ) -> None:
        super().__init__()
        self.obs_groups = list(obs_groups)
        self.obs_group_dims = tuple(int(d) for d in obs_group_dims)
        if len(self.obs_group_dims) != len(self.obs_groups):
            raise ValueError("`obs_group_dims` must match `obs_groups`.")
        self.append_obs_group = append_obs_group
        if append_obs is not None and append_obs_group not in self.obs_groups:
            raise ValueError(f"`append_obs` requires a '{append_obs_group}' observation group.")

        self.encoders = _as_group_modules(encoders, self.obs_groups, "encoders")
        self.obs_normalizers = _as_group_modules(obs_normalizers, self.obs_groups, "obs_normalizers")
        self._obs_normalizer_updates = []
        for group, normalizer in self.obs_normalizers.items():
            update = getattr(normalizer, "update", None)
            if update is not None:
                self._obs_normalizer_updates.append((group, update))
        self.append_obs = append_obs

    def forward(self, obs: TensorDict) -> torch.Tensor:
        return self.encode_groups(obs)

    def encode_groups(self, obs: TensorDict) -> torch.Tensor:
        encoded_latents: list[torch.Tensor] = []
        append_group_obs = None
        for group in self.obs_groups:
            normalized = self.obs_normalizers[group](obs[group])
            encoded_latents.append(self.encoders[group](normalized))
            if self.append_obs is not None and group == self.append_obs_group:
                append_group_obs = normalized
        latent = torch.cat(encoded_latents, dim=-1)
        if self.append_obs is None:
            return latent
        return torch.cat([latent, self.append_obs.select(append_group_obs)], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        for group, update in self._obs_normalizer_updates:
            update(obs[group])

    def as_export_module(self) -> nn.Module:
        return _GroupObsLatentAdapterExport(self)


class _ObsLatentAdapterExport(nn.Module):
    """ONNX path: input is already ``concat(obs_groups)``."""

    def __init__(self, adapter: ObsLatentAdapter) -> None:
        super().__init__()
        self._adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._adapter.encode_concat(x)


class _GroupObsLatentAdapterExport(nn.Module):
    """ONNX path: split ``concat(obs_groups)`` then reuse the training group path."""

    def __init__(self, adapter: GroupObsLatentAdapter) -> None:
        super().__init__()
        self._adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter = self._adapter
        obs = TensorDict(
            dict(zip(adapter.obs_groups, x.split(adapter.obs_group_dims, dim=-1))),
            batch_size=x.shape[:-1],
        )
        return adapter.encode_groups(obs)


def _as_group_modules(modules: Mapping[str, nn.Module], obs_groups: list[str], kind: str) -> nn.ModuleDict:
    if set(modules) != set(obs_groups):
        raise ValueError(
            f"`{kind}` must declare every obs group, got {sorted(modules)} vs obs_groups={obs_groups}."
        )
    return nn.ModuleDict({group: modules[group] for group in obs_groups})
