# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MoEModel resume, ONNX export, and pretrained expert loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import onnx
import pytest
import torch

from tests.conftest import make_obs
from z_rl.models import MLPModel, MoEModel

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def _make_moe_model(stochastic: bool = True) -> tuple[MoEModel, torch.Tensor]:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    distribution_cfg = None
    if stochastic:
        distribution_cfg = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
    model = MoEModel(
        obs,
        OBS_GROUPS,
        "actor",
        NUM_ACTIONS if stochastic else 1,
        activation="elu",
        distribution_cfg=distribution_cfg,
        num_experts=4,
        expert_hidden_dims=[16],
    )
    return model, obs


class TestMoEModelResumeAndExport:
    def test_load_state_dict_restores_deterministic_output(self) -> None:
        source, obs = _make_moe_model()
        target, _ = _make_moe_model()
        target.load_state_dict(source.state_dict())
        source.eval()
        target.eval()
        assert torch.allclose(source(obs), target(obs), atol=1e-6)

    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    def test_onnx_export_after_training_forward(self) -> None:
        """Export must succeed even when the router cache still holds a training graph."""
        actor, obs = _make_moe_model()
        actor(obs)
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


def _make_det_moe_model(num_experts: int = 4) -> tuple[MoEModel, torch.Tensor]:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    model = MoEModel(
        obs,
        OBS_GROUPS,
        "actor",
        NUM_ACTIONS,
        activation="elu",
        num_experts=num_experts,
        expert_hidden_dims=[16],
    )
    return model, obs


def _make_mlp_model() -> MLPModel:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    return MLPModel(
        obs,
        OBS_GROUPS,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[16],
        activation="elu",
    )


def _save_checkpoint(path: Path, payload: dict) -> str:
    torch.save(payload, path)
    return str(path)


def _assert_expert_equal(left: MoEModel, right: MoEModel, expert_idx: int) -> None:
    for left_w, right_w in zip(left.head.experts.weights, right.head.experts.weights):
        assert torch.equal(left_w[expert_idx], right_w[expert_idx])
    for left_b, right_b in zip(left.head.experts.biases, right.head.experts.biases):
        assert torch.equal(left_b[expert_idx], right_b[expert_idx])


def _assert_expert_not_equal(left: MoEModel, right: MoEModel, expert_idx: int) -> None:
    differs = False
    for left_w, right_w in zip(left.head.experts.weights, right.head.experts.weights):
        if not torch.equal(left_w[expert_idx], right_w[expert_idx]):
            differs = True
            break
    assert differs


class TestMoEPretrainedExpertLoading:
    def test_load_from_moe_state_dict(self, tmp_path: Path) -> None:
        source, _ = _make_det_moe_model()
        target, _ = _make_det_moe_model()
        path = _save_checkpoint(tmp_path / "moe.pt", source.state_dict())

        target.load_pretrained_experts(path)

        for idx in range(source.head.experts.num_experts):
            _assert_expert_equal(target, source, idx)

    def test_load_from_actor_state_dict_wrapper(self, tmp_path: Path) -> None:
        source, _ = _make_det_moe_model()
        target, _ = _make_det_moe_model()
        path = _save_checkpoint(tmp_path / "wrapped.pt", {"actor_state_dict": source.state_dict()})

        target.load_pretrained_experts(path)
        _assert_expert_equal(target, source, 0)

    def test_load_raw_expert_keys(self, tmp_path: Path) -> None:
        source, _ = _make_det_moe_model()
        target, _ = _make_det_moe_model()
        path = _save_checkpoint(tmp_path / "experts.pt", source.head.experts.state_dict())

        target.load_pretrained_experts(path)
        _assert_expert_equal(target, source, 1)

    def test_load_selected_expert_indices(self, tmp_path: Path) -> None:
        source, _ = _make_det_moe_model()
        target, _ = _make_det_moe_model()
        untouched = {k: v.clone() for k, v in target.head.experts.state_dict().items()}
        path = _save_checkpoint(tmp_path / "moe.pt", source.state_dict())

        target.load_pretrained_experts(path, target_expert_indices=[0, 2])

        _assert_expert_equal(target, source, 0)
        _assert_expert_equal(target, source, 2)
        for key, tensor in target.head.experts.state_dict().items():
            assert torch.equal(tensor[1], untouched[key][1])
            assert torch.equal(tensor[3], untouched[key][3])

    def test_constructor_loads_selected_index_alias(self, tmp_path: Path) -> None:
        source, obs = _make_det_moe_model()
        path = _save_checkpoint(tmp_path / "moe.pt", source.state_dict())
        target = MoEModel(
            obs,
            OBS_GROUPS,
            "actor",
            NUM_ACTIONS,
            activation="elu",
            num_experts=4,
            expert_hidden_dims=[16],
            pretrained_expert_path=path,
            pretrained_expert_target_index=1,
        )
        _assert_expert_equal(target, source, 1)
        _assert_expert_not_equal(target, source, 0)

    def test_load_mlp_head_into_all_experts(self, tmp_path: Path) -> None:
        mlp = _make_mlp_model()
        target, _ = _make_det_moe_model()
        path = _save_checkpoint(tmp_path / "mlp.pt", mlp.state_dict())

        target.load_pretrained_experts(path)

        linear_layers = [module for module in mlp.head if isinstance(module, torch.nn.Linear)]
        for layer_idx, linear in enumerate(linear_layers):
            expected_weight = linear.weight.detach().transpose(0, 1)
            expected_bias = linear.bias.detach()
            for expert_idx in range(target.head.experts.num_experts):
                assert torch.equal(target.head.experts.weights[layer_idx][expert_idx], expected_weight)
                assert torch.equal(target.head.experts.biases[layer_idx][expert_idx], expected_bias)

    def test_load_mlp_head_into_selected_expert(self, tmp_path: Path) -> None:
        mlp = _make_mlp_model()
        target, _ = _make_det_moe_model()
        untouched = target.head.experts.weights[0][1].detach().clone()
        path = _save_checkpoint(tmp_path / "mlp.pt", mlp.state_dict())

        target.load_pretrained_experts(path, target_expert_indices=[0])

        first_linear = next(module for module in mlp.head if isinstance(module, torch.nn.Linear))
        expected_weight = first_linear.weight.detach().transpose(0, 1)
        assert torch.equal(target.head.experts.weights[0][0], expected_weight)
        assert torch.equal(target.head.experts.weights[0][1], untouched)

    def test_multi_source_specs_fill_different_experts(self, tmp_path: Path) -> None:
        source_a, obs = _make_det_moe_model()
        source_b, _ = _make_det_moe_model()
        path_a = _save_checkpoint(tmp_path / "a.pt", source_a.state_dict())
        path_b = _save_checkpoint(tmp_path / "b.pt", source_b.state_dict())

        target = MoEModel(
            obs,
            OBS_GROUPS,
            "actor",
            NUM_ACTIONS,
            activation="elu",
            num_experts=4,
            expert_hidden_dims=[16],
            pretrained_expert_specs=[
                {"path": path_a, "target_expert_indices": [0]},
                {"path": path_b, "target_expert_index": 3},
            ],
        )
        _assert_expert_equal(target, source_a, 0)
        _assert_expert_equal(target, source_b, 3)

    def test_rejects_conflicting_constructor_sources(self, tmp_path: Path) -> None:
        source, obs = _make_det_moe_model()
        path = _save_checkpoint(tmp_path / "moe.pt", source.state_dict())
        with pytest.raises(ValueError, match="pretrained_expert_specs"):
            MoEModel(
                obs,
                OBS_GROUPS,
                "actor",
                NUM_ACTIONS,
                activation="elu",
                num_experts=4,
                expert_hidden_dims=[16],
                pretrained_expert_path=path,
                pretrained_expert_specs=[{"path": path}],
            )

    def test_rejects_missing_expert_keys(self, tmp_path: Path) -> None:
        target, _ = _make_det_moe_model()
        path = _save_checkpoint(tmp_path / "empty.pt", {"unrelated": torch.zeros(1)})
        with pytest.raises(KeyError, match="No expert parameters"):
            target.load_pretrained_experts(path)
