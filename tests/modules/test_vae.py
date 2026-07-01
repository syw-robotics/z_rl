# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for VAE modules."""

from __future__ import annotations

import pytest
import torch

from z_rl.modules import VAE


class TestVAE:
    """Tests for ``VAE``."""

    def test_decoder_input_dim_defaults_to_latent_dim(self) -> None:
        """The decoder should preserve the historical latent_dim input width by default."""
        vae = VAE(input_dim=8, latent_dim=5, encoder_hidden_dims=[7], decoder_hidden_dims=[6])

        assert vae.decoder_input_dim == 5
        assert vae.decoder[0].in_features == 5

    def test_decode_uses_configured_leading_latent_slice(self) -> None:
        """decode() should accept the full latent and feed only the configured slice to the decoder."""
        vae = VAE(input_dim=8, latent_dim=6, decoder_input_dim=4, encoder_hidden_dims=[7], decoder_hidden_dims=[5])
        z = torch.randn(3, 6)

        decoded = vae.decode(z)
        expected = vae.decoder(z[:, :4])

        assert decoded.shape == (3, 8)
        assert torch.allclose(decoded, expected)

    def test_forward_supports_decoder_input_dim_smaller_than_latent_dim(self) -> None:
        """A VAE forward pass should work when only part of the latent is decoded."""
        vae = VAE(input_dim=8, latent_dim=6, decoder_input_dim=4, encoder_hidden_dims=[7], decoder_hidden_dims=[5])
        x = torch.randn(3, 8)

        output = vae(x)

        assert output.shape == (3, 8)

    def test_decoder_input_dim_can_not_exceed_latent_dim(self) -> None:
        """Reject decoder input dimensions that can not be sliced from the sampled latent."""
        with pytest.raises(ValueError, match="can not exceed `latent_dim`"):
            VAE(input_dim=8, latent_dim=4, decoder_input_dim=5)
