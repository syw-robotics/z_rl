# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn

from z_rl.modules.mlp import MLP


class VAE(nn.Module):
    """Variational Autoencoder with MLP encoder and decoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        decoder_output_dim: int | None = None,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu",
    ) -> None:
        """Initialize VAE.

        Args:
            input_dim: Input feature dimension.
            latent_dim: Latent feature dimension.
            decoder_output_dim: Decoder output dimension. If ``None``, it defaults to ``input_dim``.
            encoder_hidden_dims: Hidden dimensions of encoder backbone.
            decoder_hidden_dims: Hidden dimensions of decoder backbone.
            activation: Activation function for hidden layers.
        """
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"`latent_dim` must be positive, got {latent_dim}.")
        if decoder_output_dim is not None and decoder_output_dim <= 0:
            raise ValueError(f"`decoder_output_dim` must be positive, got {decoder_output_dim}.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.decoder_output_dim = input_dim if decoder_output_dim is None else decoder_output_dim

        # Encoder backbone outputs 2 * latent_dim for [mu, log_var].
        self.encoder = MLP(input_dim, 2 * latent_dim, encoder_hidden_dims, activation)
        # Decoder maps sampled latent z back to reconstruction space.
        self.decoder = MLP(latent_dim, self.decoder_output_dim, decoder_hidden_dims, activation)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input into Gaussian posterior parameters (mu, log_var)."""
        enc_out = self.encoder(x)
        mu, log_var = torch.chunk(enc_out, 2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample latent with the reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent variable to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run VAE forward pass.

        Returns:
            z: Sampled latent variable.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        return output

    @staticmethod
    def compute_loss(
        x: torch.Tensor,
        recon_x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute VAE loss = reconstruction loss + beta * KL divergence.

        Reconstruction loss uses MSE over the last dimension.
        """
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Unsupported reduction '{reduction}'. Expected 'mean' or 'sum'.")
        if beta < 0.0:
            raise ValueError(f"`beta` must be non-negative, got {beta}.")

        recon_per_sample = torch.mean((recon_x - x).pow(2), dim=-1)
        kl_per_sample = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        if reduction == "mean":
            recon_loss = recon_per_sample.mean()
            kl_loss = kl_per_sample.mean()
        else:
            recon_loss = recon_per_sample.sum()
            kl_loss = kl_per_sample.sum()

        return recon_loss, kl_loss
