# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for VAEModel."""

from __future__ import annotations

import torch

from z_rl.models import VAEModel
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 3
OBS_GROUPS = {"actor": ["policy"]}


class TestVAEModel:
    """Tests for VAE-backed composable models."""

    def test_default_head_is_shared_decoder(self) -> None:
        """Without decoder_input_dim override, VAEModel should expose the decoder itself as the head."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        model = VAEModel(
            obs,
            OBS_GROUPS,
            "actor",
            NUM_ACTIONS,
            hidden_dims=[8],
            latent_dim=6,
            encoder_hidden_dims=[7],
            decoder_hidden_dims=[5],
        )

        assert model.head is model.vae.decoder
        assert model.vae.decoder_input_dim == 6

    def test_decoder_input_dim_uses_partial_latent_without_changing_model_latent_dim(self) -> None:
        """VAEModel should keep the full sampled latent while decoding only the configured leading slice."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        model = VAEModel(
            obs,
            OBS_GROUPS,
            "actor",
            NUM_ACTIONS,
            hidden_dims=[8],
            latent_dim=6,
            decoder_input_dim=4,
            encoder_hidden_dims=[7],
            decoder_hidden_dims=[5],
        )

        latent = model.get_latent(obs)
        head_output = model.head(latent)

        assert model.get_latent_dim() == 6
        assert latent.shape == (NUM_ENVS, 6)
        assert model.vae.decoder_input_dim == 4
        assert model.vae.decoder[0].in_features == 4
        assert head_output.shape == (NUM_ENVS, NUM_ACTIONS)
        assert torch.allclose(head_output, model.vae.decode(latent))

    def test_forward_supports_decoder_input_dim_smaller_than_latent_dim(self) -> None:
        """The model forward path should slice latents before the decoder head."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        model = VAEModel(
            obs,
            OBS_GROUPS,
            "actor",
            NUM_ACTIONS,
            hidden_dims=[8],
            latent_dim=6,
            decoder_input_dim=4,
            encoder_hidden_dims=[7],
            decoder_hidden_dims=[5],
        )

        output = model(obs)

        assert output.shape == (NUM_ENVS, NUM_ACTIONS)
