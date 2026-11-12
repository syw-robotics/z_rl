"""Tests for grouped MLP observation encoding and SimBaV2 heads."""

from __future__ import annotations

import tempfile

import onnx
import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from z_rl.models import ComposableModel, GroupMLPEncoderModel, SimBaModel
from z_rl.modules import EmpiricalNormalization, SimBa

NUM_ENVS = 4


def _make_obs() -> TensorDict:
    return TensorDict(
        {
            "proprio": torch.randn(NUM_ENVS, 6),
            "object": torch.randn(NUM_ENVS, 4),
        },
        batch_size=[NUM_ENVS],
    )


ENCODER_CFGS = {
    "proprio": {"output_dim": 5, "hidden_dims": [8]},
    "object": {"output_dim": 3, "hidden_dims": []},
}


class TestGroupMLPLatentSpec:
    def test_layer_norm_is_configured_per_encoder(self) -> None:
        model = GroupMLPEncoderModel(
            _make_obs(),
            {"actor": ["proprio", "object"]},
            "actor",
            2,
            hidden_dims=[8],
            encoder_cfgs={
                "proprio": {"output_dim": 5, "hidden_dims": [8], "layer_norm": "post_activation"},
                "object": {"output_dim": 3, "hidden_dims": [8]},
            },
        )

        assert [type(layer) for layer in model.latent_adapter.encoders["proprio"]] == [
            nn.Linear,
            nn.SiLU,
            nn.LayerNorm,
            nn.Linear,
        ]
        assert not any(isinstance(layer, nn.LayerNorm) for layer in model.latent_adapter.encoders["object"])

    def test_uses_independent_group_widths(self) -> None:
        obs = _make_obs()
        model = GroupMLPEncoderModel(
            obs,
            {"actor": ["proprio", "object"]},
            "actor",
            2,
            hidden_dims=[8],
            encoder_cfgs=ENCODER_CFGS,
            obs_normalization=True,
        )

        latent = model.get_latent(obs)
        packed = torch.cat((obs["proprio"], obs["object"]), dim=-1)

        assert latent.shape == (NUM_ENVS, 8)
        assert torch.allclose(latent, model.latent_adapter.as_export_module()(packed))
        assert set(model.latent_adapter.obs_normalizers) == {"proprio", "object"}

    def test_group_obs_normalization_overrides_model_default(self) -> None:
        obs = TensorDict(
            {
                "proprio": torch.ones(NUM_ENVS, 6) * 2,
                "object": torch.ones(NUM_ENVS, 4) * 10,
            },
            batch_size=[NUM_ENVS],
        )
        encoder_cfgs = {
            "proprio": {"output_dim": 5, "hidden_dims": [8], "obs_normalization": False},
            "object": {
                "output_dim": 3,
                "hidden_dims": [],
                "obs_normalization": {"stats_shape": (1,), "eps": 1.0e-4},
            },
        }
        model = GroupMLPEncoderModel(
            obs,
            {"actor": ["proprio", "object"]},
            "actor",
            2,
            hidden_dims=[8],
            encoder_cfgs=encoder_cfgs,
            obs_normalization=True,
        )

        proprio_normalizer = model.latent_adapter.obs_normalizers["proprio"]
        object_normalizer = model.latent_adapter.obs_normalizers["object"]
        assert isinstance(proprio_normalizer, nn.Identity)
        assert isinstance(object_normalizer, EmpiricalNormalization)
        assert tuple(object_normalizer.stats_shape) == (1,)
        assert object_normalizer.eps == 1.0e-4
        assert "obs_normalization" in encoder_cfgs["proprio"]

        model.train()
        model.latent_adapter.update_normalization(obs)
        assert object_normalizer.count == NUM_ENVS
        assert torch.allclose(object_normalizer.mean, torch.tensor(10.0))
        assert torch.allclose(proprio_normalizer(obs["proprio"]), obs["proprio"])

    def test_append_last_obs_appends_configured_group(self) -> None:
        obs = TensorDict(
            {
                "proprio": torch.arange(NUM_ENVS * 8, dtype=torch.float32).view(NUM_ENVS, 8),
                "object": torch.ones(NUM_ENVS, 4),
            },
            batch_size=[NUM_ENVS],
        )
        model = GroupMLPEncoderModel(
            obs,
            {"actor": ["proprio", "object"]},
            "actor",
            2,
            hidden_dims=[8],
            encoder_cfgs={
                "proprio": {"output_dim": 5, "hidden_dims": [8]},
                "object": {"output_dim": 3, "hidden_dims": []},
            },
            append_last_obs=True,
            append_obs_group="proprio",
            obs_group_time_slice_map={"proprio": {"last": slice(6, 8)}},
        )

        latent = model.get_latent(obs)

        assert model.latent_dim == 10
        assert latent.shape == (NUM_ENVS, 10)
        assert torch.allclose(latent[:, -2:], obs["proprio"][:, 6:8])
        exported = model.latent_adapter.as_export_module()(torch.cat([obs["proprio"], obs["object"]], dim=-1))
        assert torch.allclose(exported, latent)

    def test_append_last_obs_requires_configured_group(self) -> None:
        obs = _make_obs()
        with pytest.raises(ValueError, match="requires a 'policy' observation group"):
            GroupMLPEncoderModel(
                obs,
                {"actor": ["proprio", "object"]},
                "actor",
                2,
                hidden_dims=[8],
                encoder_cfgs=ENCODER_CFGS,
                append_last_obs=True,
            )

    def test_requires_every_active_group(self) -> None:
        obs = _make_obs()
        with pytest.raises(ValueError, match="must declare every active observation group"):
            GroupMLPEncoderModel(
                obs,
                {"actor": ["proprio", "object"]},
                "actor",
                2,
                hidden_dims=[8],
                encoder_cfgs={"proprio": ENCODER_CFGS["proprio"]},
            )


class TestSimBaHeadSpec:
    def test_composes_with_group_encoder(self) -> None:
        obs = _make_obs()
        model = ComposableModel(
            obs,
            {"actor": ["proprio", "object"]},
            "actor",
            3,
            hidden_dims=[8],
            latent_spec={"class_name": "GroupMLPLatentSpec", "encoder_cfgs": ENCODER_CFGS},
            head_spec={"class_name": "SimBaHeadSpec", "hidden_dim": 16, "num_blocks": 2, "expansion": 2},
        )

        output = model(obs)
        output.square().mean().backward()

        assert isinstance(model.head, SimBa)
        assert output.shape == (NUM_ENVS, 3)
        assert torch.isfinite(output).all()
        assert all(parameter.grad is not None for parameter in model.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    def test_named_preset_uses_simba_head(self) -> None:
        obs = _make_obs()
        model = SimBaModel(
            obs,
            {"critic": ["proprio", "object"]},
            "critic",
            1,
            hidden_dims=[8],
            hidden_dim=16,
            num_blocks=1,
            expansion=2,
        )

        assert isinstance(model.head, SimBa)
        assert model(obs).shape == (NUM_ENVS, 1)

    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning:torch.onnx")
    def test_onnx_export(self) -> None:
        obs = _make_obs()[:1]
        model = ComposableModel(
            obs,
            {"actor": ["proprio", "object"]},
            "actor",
            3,
            hidden_dims=[8],
            latent_spec={"class_name": "GroupMLPLatentSpec", "encoder_cfgs": ENCODER_CFGS},
            head_spec={"class_name": "SimBaHeadSpec", "hidden_dim": 8, "num_blocks": 1, "expansion": 2},
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()
        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),
                file.name,
                export_params=True,
                opset_version=18,
                input_names=onnx_model.input_names,
                output_names=onnx_model.output_names,
            )
            onnx.checker.check_model(onnx.load(file.name))
