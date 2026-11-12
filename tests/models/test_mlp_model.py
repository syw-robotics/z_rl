# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MLPModel."""

from __future__ import annotations

import tempfile
import torch
import torch.nn as nn
from tensordict import TensorDict

import onnx
import pytest

from tests.conftest import make_obs
from z_rl.models import (
    GroupMLPEncoderModel,
    GroupObsLatentAdapter,
    MLPEncoderModel,
    MLPModel,
    ObsLatentAdapter,
)
from z_rl.modules import EmpiricalNormalization
from z_rl.utils import ObsSelector
from z_rl.utils.checkpoint import module_state_dict_from_checkpoint

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def _make_mlp_model(stochastic: bool = False, obs_set: str = "actor", **kwargs: object) -> tuple[MLPModel, TensorDict]:
    """Create an MLPModel and matching observations for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    defaults: dict[str, object] = {"hidden_dims": [32, 32], "activation": "elu"}
    if stochastic:
        defaults["distribution_cfg"] = {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        }
    defaults.update(kwargs)
    output_dim = NUM_ACTIONS if stochastic else 1
    model = MLPModel(obs, OBS_GROUPS, obs_set, output_dim, **defaults)
    return model, obs


class TestMLPModelModes:
    """Tests for stochastic vs. deterministic forward pass behavior."""

    def test_deterministic_returns_mean(self) -> None:
        """forward() should return the distribution mean."""
        actor, obs = _make_mlp_model(stochastic=True)
        actor.eval()

        det_output = actor(obs)

        actor(obs, stochastic_output=True)
        mean_output = actor.output_mean

        assert torch.allclose(det_output, mean_output, atol=1e-6)

    def test_stochastic_differs_from_deterministic(self) -> None:
        """Stochastic samples should not exactly equal the mean (with overwhelming probability)."""
        actor, obs = _make_mlp_model(stochastic=True)

        torch.manual_seed(42)
        stochastic_output = actor(obs, stochastic_output=True)
        det_output = actor(obs, stochastic_output=False)

        assert not torch.allclose(stochastic_output, det_output, atol=1e-6), (
            "Stochastic output should differ from deterministic mean"
        )

    def test_no_distribution_returns_raw_mlp(self) -> None:
        """Without distribution_cfg, forward should return raw MLP output."""
        critic, obs = _make_mlp_model(stochastic=False, obs_set="critic")

        output = critic(obs)
        latent = critic.get_latent(obs)
        expected = critic.head(latent)
        assert torch.allclose(output, expected)


class TestMLPModelNormalization:
    """Tests for observation normalization integration."""

    def test_layer_norm_configures_hidden_layers(self) -> None:
        model, _ = _make_mlp_model(layer_norm="pre_activation")

        assert [type(layer) for layer in model.head] == [
            nn.Linear,
            nn.LayerNorm,
            nn.ELU,
            nn.Linear,
            nn.LayerNorm,
            nn.ELU,
            nn.Linear,
        ]

    def test_normalization_changes_output(self) -> None:
        """A model with obs_normalization should produce different outputs after normalization stats update."""
        model, obs = _make_mlp_model(stochastic=False, obs_set="critic", obs_normalization=True)
        model.train()

        output_before = model(obs).detach().clone()

        shifted_obs = make_obs(NUM_ENVS, OBS_DIM)
        shifted_obs["policy"] = shifted_obs["policy"] + 100.0
        for _ in range(50):
            model.update_normalization(shifted_obs)

        output_after = model(obs).detach()
        assert not torch.allclose(output_before, output_after, atol=1e-3), (
            "Output should change after normalization stats update"
        )

    def test_advanced_normalization_config_is_forwarded(self) -> None:
        """Advanced normalizer parameters should reach the model-owned normalizer."""
        model, _ = _make_mlp_model(
            stochastic=False,
            obs_set="critic",
            obs_normalization={"decay": 0.9, "stats_shape": (1,), "eps": 1.0e-4},
        )

        normalizer = model.latent_adapter.obs_normalizer
        assert isinstance(normalizer, EmpiricalNormalization)
        assert normalizer.decay == pytest.approx(0.9)
        assert normalizer.stats_shape == torch.Size((1,))
        assert normalizer.eps == pytest.approx(1.0e-4)


class TestObsGroupConcatenation:
    """Tests for observation group concatenation order."""

    def test_concatenation_order_is_preserved(self) -> None:
        """get_latent should concatenate obs groups in the declared order."""
        obs = TensorDict(
            {
                "group_a": torch.ones(2, 3),
                "group_b": torch.ones(2, 4) * 2,
            },
            batch_size=[2],
        )
        obs_groups = {"actor": ["group_a", "group_b"], "critic": ["group_a"]}
        model = MLPModel(obs, obs_groups, "actor", 1, hidden_dims=[8])

        latent = model.get_latent(obs)
        # First 3 dims should be 1.0 (group_a), next 4 should be 2.0 (group_b)
        assert torch.allclose(latent[:, :3], torch.ones(2, 3))
        assert torch.allclose(latent[:, 3:], torch.ones(2, 4) * 2)

    def test_reversed_order_gives_different_latent(self) -> None:
        """Swapping the obs group order should change the latent representation."""
        obs = TensorDict(
            {
                "group_a": torch.ones(2, 3),
                "group_b": torch.ones(2, 3) * 5,
            },
            batch_size=[2],
        )

        model_ab = MLPModel(obs, {"actor": ["group_a", "group_b"]}, "actor", 1, hidden_dims=[8])
        model_ba = MLPModel(obs, {"actor": ["group_b", "group_a"]}, "actor", 1, hidden_dims=[8])

        latent_ab = model_ab.get_latent(obs)
        latent_ba = model_ba.get_latent(obs)

        assert not torch.allclose(latent_ab, latent_ba), "Different obs group orders should produce different latents"


class TestEncoderSpec:
    """Tests for encoder-spec based latent construction."""

    def test_encoder_replaces_policy_latent(self) -> None:
        """The latent adapter should replace the normalized policy latent with the encoder output."""
        obs = TensorDict({"policy": torch.ones(2, 8) * 2}, batch_size=[2])
        model = MLPEncoderModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            1,
            hidden_dims=[8],
            encoder_latent_dim=5,
            encoder_hidden_dims=[7],
        )

        latent = model.get_latent(obs)
        encoded = model.latent_adapter.encoder(model.latent_adapter.obs_normalizer(obs["policy"]))

        assert latent.shape == (2, 5)
        assert torch.allclose(latent, encoded)

    def test_encoder_can_append_last_policy_obs(self) -> None:
        """The latent adapter should append the last policy frame when configured."""
        obs = TensorDict({"policy": torch.arange(16, dtype=torch.float32).view(2, 8)}, batch_size=[2])
        time_slice_map = {"policy": {"last": slice(6, 8)}}
        model = MLPEncoderModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            1,
            hidden_dims=[8],
            encoder_latent_dim=5,
            encoder_hidden_dims=[7],
            append_last_obs=True,
            obs_group_time_slice_map=time_slice_map,
        )

        latent = model.get_latent(obs)
        normalized = model.latent_adapter.obs_normalizer(obs["policy"])
        encoded = model.latent_adapter.encoder(normalized)

        assert latent.shape == (2, 7)
        assert torch.allclose(latent[:, :5], encoded)
        assert torch.allclose(latent[:, 5:], obs["policy"][:, 6:8])

    def test_encoder_append_obs_group_must_exist(self) -> None:
        """append_last_obs should reject an append group that is not the active policy group."""
        obs = TensorDict({"policy": torch.ones(2, 8)}, batch_size=[2])

        with pytest.raises(ValueError, match="exactly one active observation group named 'prop'"):
            MLPEncoderModel(
                obs,
                {"actor": ["policy"]},
                "actor",
                1,
                hidden_dims=[8],
                encoder_latent_dim=4,
                encoder_hidden_dims=[6],
                append_last_obs=True,
                append_obs_group="prop",
            )

    def test_encoder_requires_policy_only_obs_group(self) -> None:
        """Encoder specs should reject non-policy or multi-group observation sets."""
        obs = TensorDict({"group_a": torch.ones(2, 3), "policy": torch.ones(2, 2)}, batch_size=[2])

        with pytest.raises(ValueError, match="exactly one active observation group named 'policy'"):
            MLPEncoderModel(
                obs,
                {"actor": ["group_a", "policy"]},
                "actor",
                1,
                hidden_dims=[8],
                encoder_latent_dim=4,
                encoder_hidden_dims=[6],
            )


class TestMLPModelExport:
    """Tests for ONNX export."""

    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    def test_onnx_export_model(self) -> None:
        """ONNX-exported MLP model should be a valid ONNX graph with correct I/O names."""
        actor, _obs = _make_mlp_model(stochastic=True)
        actor.eval()

        onnx_model = actor.as_onnx(verbose=False)
        onnx_model.eval()

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

            assert [i.name for i in loaded.graph.input] == ["obs"]
            assert [o.name for o in loaded.graph.output] == ["actions"]

    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    def test_onnx_export_with_normalization(self) -> None:
        """ONNX export should produce a valid graph when obs normalization is enabled."""
        model, _obs = _make_mlp_model(stochastic=True, obs_normalization=True)
        model.train()

        for _ in range(50):
            shifted_obs = TensorDict(
                {"policy": torch.randn(NUM_ENVS, OBS_DIM) + 5.0},
                batch_size=[NUM_ENVS],
            )
            model.update_normalization(shifted_obs)

        model.eval()
        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()

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

    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    def test_onnx_export_with_encoder_spec(self) -> None:
        """ONNX export should include encoder-spec latent adapters without custom wrappers."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        model = MLPEncoderModel(
            obs,
            OBS_GROUPS,
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 32],
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
            encoder_latent_dim=6,
            encoder_hidden_dims=[12],
            append_last_obs=True,
            obs_group_time_slice_map={"policy": {"last": slice(6, 8)}},
        )
        model.eval()

        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()

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


class TestObsLatentAdapter:
    """Concat groups, one normalizer, one encoder."""

    def test_flat_tensor_matches_tensordict(self) -> None:
        obs = TensorDict({"a": torch.randn(2, 3), "b": torch.randn(2, 4)}, batch_size=[2])
        adapter = ObsLatentAdapter(
            obs_groups=["a", "b"],
            obs_normalizer=nn.Identity(),
            encoder=nn.Linear(7, 5),
        )
        concat = torch.cat([obs["a"], obs["b"]], dim=-1)
        assert torch.allclose(adapter(obs), adapter.as_export_module()(concat))

    def test_append_obs_selects_from_concat(self) -> None:
        obs = TensorDict({"policy": torch.arange(16, dtype=torch.float32).view(2, 8)}, batch_size=[2])
        adapter = ObsLatentAdapter(
            obs_groups=["policy"],
            obs_normalizer=nn.Identity(),
            append_obs=ObsSelector(slice(6, 8)),
        )
        latent = adapter(obs)
        assert torch.allclose(latent[:, :8], obs["policy"])
        assert torch.allclose(latent[:, 8:], obs["policy"][:, 6:8])

class TestGroupObsLatentAdapter:
    """Per-group normalizer and encoder, then concat."""

    def test_encodes_each_group_then_cats(self) -> None:
        obs = TensorDict({"prop": torch.randn(2, 3), "scan": torch.randn(2, 4)}, batch_size=[2])
        adapter = GroupObsLatentAdapter(
            obs_groups=["prop", "scan"],
            obs_group_dims=(3, 4),
            encoders={"prop": nn.Linear(3, 2), "scan": nn.Linear(4, 5)},
            obs_normalizers={"prop": nn.Identity(), "scan": nn.Identity()},
        )
        concat = torch.cat([obs["prop"], obs["scan"]], dim=-1)
        latent = adapter(obs)
        assert latent.shape == (2, 7)
        assert torch.allclose(latent, adapter.as_export_module()(concat))

    def test_update_normalization_is_per_group(self) -> None:
        adapter = GroupObsLatentAdapter(
            obs_groups=["prop", "scan"],
            obs_group_dims=(3, 4),
            encoders={"prop": nn.Identity(), "scan": nn.Identity()},
            obs_normalizers={"prop": EmpiricalNormalization(3), "scan": EmpiricalNormalization(4)},
        )
        obs = TensorDict(
            {"prop": torch.ones(8, 3) * 2, "scan": torch.ones(8, 4) * 10},
            batch_size=[8],
        )
        adapter.train()
        adapter.update_normalization(obs)
        assert adapter.obs_normalizers["prop"].count == 8
        assert adapter.obs_normalizers["scan"].count == 8
        assert torch.allclose(adapter.obs_normalizers["prop"].mean, torch.full((3,), 2.0))
        assert torch.allclose(adapter.obs_normalizers["scan"].mean, torch.full((4,), 10.0))

    def test_append_obs_appends_last_policy_frame(self) -> None:
        obs = TensorDict(
            {"policy": torch.arange(16, dtype=torch.float32).view(2, 8), "scan": torch.ones(2, 4)},
            batch_size=[2],
        )
        adapter = GroupObsLatentAdapter(
            obs_groups=["policy", "scan"],
            obs_group_dims=(8, 4),
            encoders={"policy": nn.Identity(), "scan": nn.Identity()},
            obs_normalizers={"policy": nn.Identity(), "scan": nn.Identity()},
            append_obs=ObsSelector(slice(6, 8)),
        )
        latent = adapter(obs)
        assert latent.shape == (2, 14)
        assert torch.allclose(latent[:, :8], obs["policy"])
        assert torch.allclose(latent[:, 8:12], obs["scan"])
        assert torch.allclose(latent[:, 12:], obs["policy"][:, 6:8])

    def test_append_obs_uses_configured_group(self) -> None:
        obs = TensorDict(
            {"prop": torch.arange(16, dtype=torch.float32).view(2, 8), "scan": torch.ones(2, 4)},
            batch_size=[2],
        )
        adapter = GroupObsLatentAdapter(
            obs_groups=["prop", "scan"],
            obs_group_dims=(8, 4),
            encoders={"prop": nn.Identity(), "scan": nn.Identity()},
            obs_normalizers={"prop": nn.Identity(), "scan": nn.Identity()},
            append_obs=ObsSelector(slice(6, 8)),
            append_obs_group="prop",
        )
        latent = adapter(obs)
        assert adapter.append_obs_group == "prop"
        assert torch.allclose(latent[:, 12:], obs["prop"][:, 6:8])
        exported = adapter.as_export_module()(torch.cat([obs["prop"], obs["scan"]], dim=-1))
        assert torch.allclose(exported, latent)

    def test_encoders_must_cover_every_group(self) -> None:
        with pytest.raises(ValueError, match="encoders"):
            GroupObsLatentAdapter(
                obs_groups=["prop", "scan"],
                obs_group_dims=(3, 4),
                encoders={"prop": nn.Identity()},
                obs_normalizers={"prop": nn.Identity(), "scan": nn.Identity()},
            )


class TestModuleCheckpoints:
    """Per-child checkpoint entries used to initialize one module from another model."""

    def test_group_encoder_modules_round_trip_without_touching_the_head(self) -> None:
        obs = TensorDict(
            {"proprio": torch.randn(NUM_ENVS, 6), "object": torch.randn(NUM_ENVS, 4)},
            batch_size=[NUM_ENVS],
        )
        encoder_cfgs = {
            "proprio": {"output_dim": 5, "hidden_dims": [8]},
            "object": {"output_dim": 3, "hidden_dims": []},
        }
        teacher = GroupMLPEncoderModel(
            obs, {"actor": ["proprio", "object"]}, "actor", 2, hidden_dims=[8], encoder_cfgs=encoder_cfgs
        )
        student = GroupMLPEncoderModel(
            obs, {"actor": ["proprio", "object"]}, "actor", 2, hidden_dims=[8], encoder_cfgs=encoder_cfgs
        )
        student_head = {key: value.clone() for key, value in student.head.state_dict().items()}

        modules = teacher.module_state_dicts()
        student.load_module_state_dict("latent_adapter", modules["latent_adapter"])

        assert set(modules) == {"latent_adapter", "head"}
        assert "encoders.proprio.0.weight" in modules["latent_adapter"]
        assert "encoders.object.weight" in modules["latent_adapter"]
        assert torch.equal(
            student.latent_adapter.encoders["proprio"][0].weight,
            teacher.latent_adapter.encoders["proprio"][0].weight,
        )
        assert all(torch.equal(student.head.state_dict()[key], student_head[key]) for key in student_head)

    def test_old_flat_checkpoint_and_teacher_role_fallback(self) -> None:
        teacher, _obs = _make_mlp_model(stochastic=True)
        checkpoint = {"actor_state_dict": teacher.state_dict()}

        head = module_state_dict_from_checkpoint(checkpoint, "head", role="teacher")

        assert torch.equal(head["0.weight"], teacher.head.state_dict()["0.weight"])
        with pytest.raises(KeyError, match="not found"):
            module_state_dict_from_checkpoint(checkpoint, "rnn", role="actor")
