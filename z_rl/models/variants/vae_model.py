# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from z_rl.modules import VAE

from z_rl.models.composition import ComposableModel, HeadSpec, LatentSpec


@dataclass(slots=True)
class VAELatentSpec(LatentSpec):
    """Latent spec that replaces normalized policy observations with sampled VAE latent."""

    latent_dim: int = 64
    encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256)
    decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256)
    activation: str = "elu"

    def validate(self, model: nn.Module) -> None:
        """Validate assumptions needed by the VAE latent/head composition."""
        if getattr(model, "obs_groups", None) != ["policy"]:
            raise ValueError(
                "`VAELatentSpec` requires exactly one active observation group named 'policy'. "
                f"Got {getattr(model, 'obs_groups', None)}."
            )
        if self.latent_dim <= 0:
            raise ValueError(f"`latent_dim` must be positive, got {self.latent_dim}.")
        if isinstance(self.encoder_hidden_dims, (tuple, list)) and len(self.encoder_hidden_dims) == 0:
            raise ValueError("`encoder_hidden_dims` can not be empty.")
        if isinstance(self.decoder_hidden_dims, (tuple, list)) and len(self.decoder_hidden_dims) == 0:
            raise ValueError("`decoder_hidden_dims` can not be empty.")

    def build_latent_adapter(self, model: nn.Module) -> nn.Module:
        """Build a VAE whose encoder is used as latent adapter and decoder as model head."""
        vae = VAE(
            input_dim=model.obs_dim,
            latent_dim=self.latent_dim,
            decoder_output_dim=getattr(model, "_vae_decoder_output_dim"),
            encoder_hidden_dims=self.encoder_hidden_dims,
            decoder_hidden_dims=self.decoder_hidden_dims,
            activation=self.activation,
        )
        return _VAELatentAdapter(vae=vae)

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
        if not isinstance(output_dim, int):
            raise TypeError(
                "`VAEModel` only supports integer decoder output dimensions. "
                f"Got output_dim={output_dim}."
            )
        decoder = model.latent_adapter.vae.decoder
        if getattr(model, "_vae_decoder_output_dim", None) != output_dim:
            raise ValueError(
                f"VAE decoder output_dim mismatch: expected {output_dim}, "
                f"got {getattr(model, '_vae_decoder_output_dim', None)}."
            )
        return decoder


class _VAELatentAdapter(nn.Module):
    """Latent adapter that performs VAE encode + reparameterize."""

    def __init__(self, vae: VAE) -> None:
        super().__init__()
        self.vae = vae
        self.last_mu: torch.Tensor | None = None
        self.last_log_var: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` and sample latent ``z`` using the reparameterization trick."""
        mu, log_var = self.vae.encode(x)
        z = self.vae.reparameterize(mu, log_var)
        self.last_mu = mu
        self.last_log_var = log_var
        return z


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
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        latent_dim: int = 64,
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
            latent_spec=VAELatentSpec(
                latent_dim=latent_dim,
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
