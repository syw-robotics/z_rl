# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from z_rl.modules import MLP, VAE

from z_rl.models.composition import ComposableModel, HeadSpec, LatentSpec
from z_rl.models.mlp_model import ObservationNormalizationConfig


@dataclass(slots=True)
class VAEEncoderLatentSpec(LatentSpec):
    """Latent spec that replaces normalized policy observations with sampled VAE latent."""

    latent_dim: int = 64
    decoder_input_dim: int | None = None
    encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256)
    decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256)
    activation: str = "elu"

    def validate(self, model: nn.Module) -> None:
        """Validate assumptions needed by the VAE latent/head composition."""
        if getattr(model, "obs_groups", None) != ["policy"]:
            raise ValueError(
                "`VAEEncoderLatentSpec` requires exactly one active observation group named 'policy'. "
                f"Got {getattr(model, 'obs_groups', None)}."
            )
        if self.latent_dim <= 0:
            raise ValueError(f"`latent_dim` must be positive, got {self.latent_dim}.")
        if self.decoder_input_dim is not None:
            if self.decoder_input_dim <= 0:
                raise ValueError(f"`decoder_input_dim` must be positive, got {self.decoder_input_dim}.")
            if self.decoder_input_dim > self.latent_dim:
                raise ValueError(
                    f"`decoder_input_dim` can not exceed `latent_dim`, "
                    f"got {self.decoder_input_dim} > {self.latent_dim}."
                )
        if isinstance(self.encoder_hidden_dims, (tuple, list)) and len(self.encoder_hidden_dims) == 0:
            raise ValueError("`encoder_hidden_dims` can not be empty.")
        if isinstance(self.decoder_hidden_dims, (tuple, list)) and len(self.decoder_hidden_dims) == 0:
            raise ValueError("`decoder_hidden_dims` can not be empty.")

    def build_latent_adapter(self, model: nn.Module) -> nn.Module:
        """Build a VAE whose encoder is used as latent adapter and decoder as model head."""
        vae = VAE(
            input_dim=model.obs_dim,
            latent_dim=self.latent_dim,
            decoder_input_dim=self.decoder_input_dim,
            decoder_output_dim=getattr(model, "_vae_decoder_output_dim"),
            encoder_hidden_dims=self.encoder_hidden_dims,
            decoder_hidden_dims=self.decoder_hidden_dims,
            activation=self.activation,
        )
        return _VAELatentAdapter(
            obs_group="policy",
            obs_normalizer=model._build_obs_normalizer(model.obs_normalization),
            vae=vae,
        )

    def get_latent_dim(self, model: nn.Module) -> int:
        """Return sampled latent width."""
        return self.latent_dim


@dataclass(slots=True)
class VAEDecoderHeadSpec(HeadSpec):
    """Head spec that reuses the decoder from the latent adapter VAE."""

    def validate(self, model: nn.Module) -> None:
        """Ensure the latent adapter is a VAE adapter before wiring the head."""
        if not isinstance(getattr(model, "latent_adapter", None), _VAELatentAdapter):
            raise TypeError(
                "`VAEDecoderHeadSpec` requires model.latent_adapter to be `_VAELatentAdapter`, "
                f"got {type(getattr(model, 'latent_adapter', None))}."
            )

    def build_head(self, model: nn.Module, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        """Return the decoder from the same VAE used by the latent adapter."""
        return model.latent_adapter.vae.decoder


class _VAELatentAdapter(nn.Module):
    """Latent adapter that performs VAE encode + reparameterize."""

    def __init__(self, obs_group: str, obs_normalizer: nn.Module, vae: VAE) -> None:
        super().__init__()
        self.obs_group = obs_group
        self.obs_normalizer = obs_normalizer
        self.vae = vae
        self.last_mu: torch.Tensor | None = None
        self.last_log_var: torch.Tensor | None = None

    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Normalize and encode the configured observation group."""
        x = self.obs_normalizer(obs[self.obs_group])
        mu, log_var = self.vae.encode(x)
        z = self.vae.reparameterize(mu, log_var)
        self.last_mu = mu
        self.last_log_var = log_var
        return z

    def update_normalization(self, obs: TensorDict) -> None:
        """Update running normalization statistics for the encoded observation group."""
        update = getattr(self.obs_normalizer, "update", None)
        if update is not None:
            update(obs[self.obs_group])

    def as_export_module(self) -> nn.Module:
        """Return a tensor-only adapter for ONNX export."""
        return _VAELatentAdapterExporter(self.obs_normalizer, self.vae)


class _VAELatentAdapterExporter(nn.Module):
    """Tensor-only export adapter for VAE latents."""

    def __init__(self, obs_normalizer: nn.Module, vae: VAE) -> None:
        super().__init__()
        self.obs_normalizer = obs_normalizer
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize, encode, and sample a VAE latent."""
        x = self.obs_normalizer(x)
        mu, log_var = self.vae.encode(x)
        return self.vae.reparameterize(mu, log_var)


class VAEModel(ComposableModel):
    """Composable MLP variant with VAE latent and decoder head.

    Data flow: ``policy obs -> (normalization) -> VAE encoder -> reparameterize -> VAE decoder -> output``.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: ObservationNormalizationConfig = False,
        distribution_cfg: dict | None = None,
        latent_dim: int = 64,
        decoder_input_dim: int | None = None,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        vae_activation: str = "elu",
    ) -> None:
        """Initialize VAE-backed model where head is equivalent to decoder."""
        self._vae_decoder_output_dim = output_dim
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            latent_spec=VAEEncoderLatentSpec(
                latent_dim=latent_dim,
                decoder_input_dim=decoder_input_dim,
                encoder_hidden_dims=encoder_hidden_dims,
                decoder_hidden_dims=decoder_hidden_dims,
                activation=vae_activation,
            ),
            head_spec=VAEDecoderHeadSpec(),
        )

    @property
    def vae(self) -> VAE:
        """Expose the shared VAE instance used by latent adapter and decoder head."""
        return self.latent_adapter.vae
