# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for CNNLatentSpec and CNNModel."""

from __future__ import annotations

import tempfile

import torch
from tensordict import TensorDict

import onnx
import pytest

from z_rl.models import CNNModel, ComposableModel, GroupObsLatentAdapter, MLPModel

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
IMG_C, IMG_H, IMG_W = 1, 16, 16
CNN_CFG = {"output_channels": [4], "kernel_size": 3, "stride": 2}


def _make_cnn_obs(*, extra_image: bool = False) -> TensorDict:
    data: dict[str, torch.Tensor] = {
        "policy": torch.randn(NUM_ENVS, OBS_DIM),
        "image": torch.randn(NUM_ENVS, IMG_C, IMG_H, IMG_W),
    }
    if extra_image:
        data["depth"] = torch.randn(NUM_ENVS, IMG_C, IMG_H, IMG_W)
    return TensorDict(data, batch_size=[NUM_ENVS])


def _cnn_output_dim() -> int:
    from z_rl.modules import CNN

    cnn = CNN(input_dim=(IMG_H, IMG_W), input_channels=IMG_C, **CNN_CFG)
    return int(cnn.output_dim)


class TestCNNLatentSpec:
    def test_mixed_groups_concat_vector_and_cnn(self) -> None:
        obs = _make_cnn_obs()
        model = CNNModel(
            obs,
            {"actor": ["policy", "image"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[16],
            image_obs_group="image",
            cnn_cfg=CNN_CFG,
        )
        latent = model.get_latent(obs)
        expected_dim = OBS_DIM + _cnn_output_dim()
        assert latent.shape == (NUM_ENVS, expected_dim)
        assert model(obs).shape == (NUM_ENVS, NUM_ACTIONS)
        assert isinstance(model.latent_adapter, GroupObsLatentAdapter)
        assert model.latent_adapter.image_obs_group == "image"
        assert set(model.latent_adapter.encoders) == {"policy", "image"}

    def test_rejects_image_only_obs(self) -> None:
        obs = TensorDict({"image": torch.randn(NUM_ENVS, IMG_C, IMG_H, IMG_W)}, batch_size=[NUM_ENVS])
        with pytest.raises(ValueError, match="at least one 1D observation group"):
            CNNModel(
                obs,
                {"actor": ["image"]},
                "actor",
                NUM_ACTIONS,
                hidden_dims=[16],
                image_obs_group="image",
                cnn_cfg=CNN_CFG,
            )

    def test_append_last_obs_appends_last_policy_frame(self) -> None:
        obs = _make_cnn_obs()
        obs["policy"] = torch.arange(NUM_ENVS * OBS_DIM, dtype=torch.float32).view(NUM_ENVS, OBS_DIM)
        model = CNNModel(
            obs,
            {"actor": ["policy", "image"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[16],
            image_obs_group="image",
            cnn_cfg=CNN_CFG,
            append_last_obs=True,
            obs_group_time_slice_map={"policy": {"last": slice(6, 8)}},
        )
        latent = model.get_latent(obs)
        expected_dim = OBS_DIM + _cnn_output_dim() + 2
        assert latent.shape == (NUM_ENVS, expected_dim)
        assert torch.allclose(latent[:, -2:], obs["policy"][:, 6:8])
        assert model.latent_adapter.append_obs is not None

    def test_append_last_obs_uses_configured_group(self) -> None:
        obs = TensorDict(
            {
                "prop": torch.arange(NUM_ENVS * OBS_DIM, dtype=torch.float32).view(NUM_ENVS, OBS_DIM),
                "image": torch.randn(NUM_ENVS, IMG_C, IMG_H, IMG_W),
            },
            batch_size=[NUM_ENVS],
        )
        model = CNNModel(
            obs,
            {"actor": ["prop", "image"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[16],
            image_obs_group="image",
            cnn_cfg=CNN_CFG,
            append_last_obs=True,
            append_obs_group="prop",
            obs_group_time_slice_map={"prop": {"last": slice(6, 8)}},
        )
        latent = model.get_latent(obs)
        assert latent.shape == (NUM_ENVS, OBS_DIM + _cnn_output_dim() + 2)
        assert torch.allclose(latent[:, -2:], obs["prop"][:, 6:8])
        assert model.latent_adapter.append_obs_group == "prop"
        exported = model.latent_adapter.as_export_module()(obs["prop"], obs["image"])
        assert torch.allclose(exported, latent)

    def test_append_last_obs_requires_policy_group(self) -> None:
        obs = TensorDict(
            {
                "prop": torch.randn(NUM_ENVS, OBS_DIM),
                "image": torch.randn(NUM_ENVS, IMG_C, IMG_H, IMG_W),
            },
            batch_size=[NUM_ENVS],
        )
        with pytest.raises(ValueError, match="requires a 'policy' observation group"):
            CNNModel(
                obs,
                {"actor": ["prop", "image"]},
                "actor",
                NUM_ACTIONS,
                hidden_dims=[16],
                image_obs_group="image",
                cnn_cfg=CNN_CFG,
                append_last_obs=True,
                obs_group_time_slice_map={"policy": {"last": slice(6, 8)}},
            )

    def test_projection_sets_cnn_width(self) -> None:
        obs = _make_cnn_obs()
        model = CNNModel(
            obs,
            {"actor": ["policy", "image"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[16],
            image_obs_group="image",
            cnn_cfg=CNN_CFG,
            cnn_projection_cfg={"output_dim": 12, "hidden_dims": []},
        )
        latent = model.get_latent(obs)
        assert latent.shape == (NUM_ENVS, OBS_DIM + 12)

    def test_composable_model_accepts_spec_config(self) -> None:
        obs = _make_cnn_obs()
        model = ComposableModel(
            obs,
            {"actor": ["policy", "image"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[16],
            latent_spec={"class_name": "CNNLatentSpec", "cnn_cfg": CNN_CFG, "image_obs_group": "image"},
            head_spec={"class_name": "MoEHeadSpec", "num_experts": 2, "expert_hidden_dims": [8]},
        )
        output = model(obs)
        assert model.head.num_experts == 2
        assert output.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_rejects_extra_image_group(self) -> None:
        obs = _make_cnn_obs(extra_image=True)
        with pytest.raises(ValueError, match="Non-CNN observation groups must be 1D"):
            CNNModel(
                obs,
                {"actor": ["policy", "image", "depth"]},
                "actor",
                NUM_ACTIONS,
                hidden_dims=[16],
                image_obs_group="image",
                cnn_cfg=CNN_CFG,
            )

    def test_requires_image_obs_group(self) -> None:
        obs = _make_cnn_obs()
        with pytest.raises(ValueError, match="must be one of"):
            CNNModel(
                obs,
                {"actor": ["policy", "image"]},
                "actor",
                NUM_ACTIONS,
                hidden_dims=[16],
                cnn_cfg=CNN_CFG,
            )

    def test_rejects_per_group_cnn_cfg(self) -> None:
        obs = _make_cnn_obs()
        with pytest.raises(ValueError, match="single `cnn_cfg` dict"):
            CNNModel(
                obs,
                {"actor": ["policy", "image"]},
                "actor",
                NUM_ACTIONS,
                hidden_dims=[16],
                image_obs_group="image",
                cnn_cfg={"image": CNN_CFG},
            )

    def test_mlp_model_still_rejects_image_groups(self) -> None:
        obs = _make_cnn_obs()
        with pytest.raises(ValueError, match="only supports 1D observations"):
            MLPModel(
                obs,
                {"actor": ["policy", "image"]},
                "actor",
                NUM_ACTIONS,
                hidden_dims=[16],
            )


class TestCNNModelONNXExport:
    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning:torch.onnx")
    def test_onnx_export_mixed_groups(self) -> None:
        obs = _make_cnn_obs()
        obs = TensorDict(
            {
                "policy": obs["policy"][:1],
                "image": obs["image"][:1],
            },
            batch_size=[1],
        )
        model = CNNModel(
            obs,
            {"actor": ["policy", "image"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[16],
            image_obs_group="image",
            cnn_cfg=CNN_CFG,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()
        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()
        assert onnx_model.input_names == ["obs", "image"]

        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),
                f.name,
                export_params=True,
                opset_version=18,
                input_names=onnx_model.input_names,
                output_names=onnx_model.output_names,
            )
            loaded = onnx.load(f.name)
            onnx.checker.check_model(loaded)
            assert [i.name for i in loaded.graph.input] == onnx_model.input_names
            assert [o.name for o in loaded.graph.output] == onnx_model.output_names


class TestCNNLeadingDims:
    def test_cnn_preserves_time_major_batch(self) -> None:
        from z_rl.modules import CNN

        cnn = CNN(input_dim=(IMG_H, IMG_W), input_channels=IMG_C, **CNN_CFG)
        time_steps, num_envs = 3, 4
        images = torch.randn(time_steps, num_envs, IMG_C, IMG_H, IMG_W)
        output = cnn(images)
        assert output.shape == (time_steps, num_envs, int(cnn.output_dim))
