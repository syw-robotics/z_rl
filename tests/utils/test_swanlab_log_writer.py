# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the SwanLab writer and its logger wiring."""

from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from z_rl.utils.logger import Logger
from z_rl.utils.log_writers import SwanLabLogWriter


def _logger(log_dir: str, logger_cfg: str | dict, swanlab_project: str = "demo") -> Logger:
    return Logger(
        log_dir=log_dir,
        cfg={"logger": logger_cfg, "swanlab_project": swanlab_project, "num_steps_per_env": 4},
        env_cfg={},
        num_envs=2,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )


def test_init_mirrors_wandb_run_setup(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SwanLab should open a named run from the log directory, like W&B."""
    monkeypatch.delenv("SWANLAB_WORKSPACE", raising=False)
    fake = MagicMock()
    monkeypatch.setattr("z_rl.utils.log_writers.swanlab", fake)
    log_dir = str(tmp_path / "2026-10-13_run")

    with patch("torch.utils.tensorboard.SummaryWriter") as summary_cls:
        SwanLabLogWriter(log_dir=log_dir, project_name="demo")

    summary_cls.assert_called_once_with(log_dir, flush_secs=10)
    fake.init.assert_called_once_with(
        project="demo",
        workspace=None,
        experiment_name="2026-10-13_run",
        config={"log_dir": log_dir},
    )


def test_workspace_comes_from_environment(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SWANLAB_WORKSPACE should fill the same role as WANDB_USERNAME."""
    monkeypatch.setenv("SWANLAB_WORKSPACE", "lab")
    fake = MagicMock()
    monkeypatch.setattr("z_rl.utils.log_writers.swanlab", fake)

    with patch("torch.utils.tensorboard.SummaryWriter"):
        SwanLabLogWriter(log_dir=str(tmp_path), cfg={"swanlab_project": "legacy"})

    assert fake.init.call_args.kwargs["project"] == "legacy"
    assert fake.init.call_args.kwargs["workspace"] == "lab"


def test_missing_package_and_project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction should fail with the same kinds of errors as the W&B writer."""
    monkeypatch.setattr("z_rl.utils.log_writers.swanlab", None)
    with pytest.raises(ModuleNotFoundError, match="swanlab"):
        SwanLabLogWriter(log_dir=str(tmp_path), project_name="demo")

    monkeypatch.setattr("z_rl.utils.log_writers.swanlab", MagicMock())
    with patch("torch.utils.tensorboard.SummaryWriter"), pytest.raises(KeyError, match="project_name"):
        SwanLabLogWriter(log_dir=str(tmp_path))


def test_store_config_scalars_and_artifacts(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config, scalars, checkpoints, files, and videos should reach SwanLab."""
    fake = MagicMock()
    monkeypatch.setattr("z_rl.utils.log_writers.swanlab", fake)

    @dataclass
    class EnvCfg:
        size: int = 3

    class EnvWithDict:
        def to_dict(self) -> dict:
            return {"size": 1}

    with patch("torch.utils.tensorboard.SummaryWriter"):
        writer = SwanLabLogWriter(log_dir=str(tmp_path), project_name="demo")
        writer.store_config(EnvWithDict(), {"lr": 1e-3})
        writer.store_config(EnvCfg(), {"lr": 1e-3})
        writer.add_scalar("Loss/value", torch.tensor(0.25), 4)
        writer.save_model(str(tmp_path / "model_4.pt"), 4)
        writer.save_file(str(tmp_path / "git" / "z_rl.diff"))
        gif = tmp_path / "clip.gif"
        mp4 = tmp_path / "clip.mp4"
        writer.save_video(gif, 4)
        writer.save_video(mp4, 4)
        writer.save_video(mp4, 5)
        writer.stop()

    assert fake.config.update.call_args_list[0].args == ({"train_cfg": {"lr": 1e-3}},)
    assert fake.config.update.call_args_list[1].args == ({"env_cfg": {"size": 1}},)
    assert fake.config.update.call_args_list[3].args == ({"env_cfg": {"size": 3}},)
    fake.log.assert_any_call({"Loss/value": 0.25}, step=4)
    fake.log.assert_any_call({"video": fake.Video.return_value}, step=4)
    fake.Video.assert_called_once_with(str(gif), caption="clip.gif")
    fake.save.assert_any_call(str(tmp_path / "model_4.pt"), base_path=str(tmp_path), policy="now")
    fake.save.assert_any_call(str(tmp_path / "git" / "z_rl.diff"), base_path=str(tmp_path / "git"), policy="now")
    fake.save.assert_any_call(str(mp4), base_path=str(tmp_path), policy="now")
    assert fake.save.call_count == 3
    writer.tensorboard_writer.close.assert_called_once()
    fake.finish.assert_called_once()


def test_logger_accepts_swanlab_like_wandb(tmp_path: pathlib.Path) -> None:
    """String and dict logger configs should both construct SwanLabLogWriter."""
    created: dict[str, str | None] = {}

    class FakeWriter:
        def __init__(self, log_dir: str, project_name: str | None = None, **kwargs: object) -> None:
            created["log_dir"] = log_dir
            created["project_name"] = project_name

    with patch("z_rl.utils.SwanLabLogWriter", FakeWriter):
        string_logger = _logger(str(tmp_path / "string"), "swanlab", swanlab_project="from-string")
        with pytest.warns(DeprecationWarning, match="SwanLabLogWriter"):
            string_logger.init_logging_writer()
        assert string_logger.logger_type == "SwanLabLogWriter"
        assert created == {"log_dir": str(tmp_path / "string"), "project_name": "from-string"}

        dict_logger = _logger(str(tmp_path / "dict"), {"class_name": "SwanLabLogWriter", "project_name": "from-dict"})
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            dict_logger.init_logging_writer()
        assert dict_logger.logger_type == "SwanLabLogWriter"
        assert created["project_name"] == "from-dict"


def test_logger_skips_time_axis_metrics_for_swanlab(tmp_path: pathlib.Path) -> None:
    """SwanLab should omit wall-clock series, matching the W&B writer."""
    logger = _logger(str(tmp_path), "tensorboard")
    logger.logger_type = "SwanLabLogWriter"
    logger.writer = MagicMock()
    logger.rewbuffer.append(1.5)
    logger.lenbuffer.append(8)

    logger.log(
        it=1,
        start_it=0,
        total_it=2,
        collect_time=0.2,
        learn_time=0.3,
        loss_dict={"value": 0.1},
        learning_rate=1e-3,
        print_minimal=True,
    )

    tags = [call.args[0] for call in logger.writer.add_scalar.call_args_list]
    assert "Train/mean_reward" in tags
    assert "Train/mean_episode_length" in tags
    assert "Train/mean_reward/time" not in tags
    assert "Train/mean_episode_length/time" not in tags
