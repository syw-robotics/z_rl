# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Logging backends.

``LogWriter`` is the interface used by ``Logger``. WandB and SwanLab share the remote-run
flow; Neptune keeps its own upload API. Each remote writer also keeps a local TensorBoard log.
"""

from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod
from dataclasses import asdict


class LogWriter(ABC):
    """Optional interface for logging backends that can upload artifacts."""

    @abstractmethod
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Log a scalar metric."""

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload configuration at run start."""

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint."""

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file."""

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video file."""

    def stop(self) -> None:
        """Finalize the logging run."""


try:
    import wandb
except ModuleNotFoundError:
    wandb = None

try:
    import swanlab
except ModuleNotFoundError:
    swanlab = None

try:
    import neptune
except ModuleNotFoundError:
    neptune = None


class _RemoteLogWriter(LogWriter):
    """TensorBoard plus one remote run with shared config, scalar, and file upload."""

    service_name: str
    project_cfg_key: str

    def __init__(
        self,
        log_dir: str,
        project_name: str | None = None,
        flush_secs: int = 10,
        cfg: dict | None = None,
    ) -> None:
        """Open TensorBoard and the remote run."""
        if self._client() is None:
            raise ModuleNotFoundError(self._missing_package_message())
        from torch.utils.tensorboard import SummaryWriter

        self.tensorboard_writer = SummaryWriter(log_dir, flush_secs=flush_secs)
        project = project_name or (cfg or {}).get(self.project_cfg_key)
        if project is None:
            raise KeyError(f"Please specify {self.service_name} project_name in logger config.")
        self.logged_videos: set[str] = set()
        self._init_run(log_dir, project, os.path.split(log_dir)[-1])

    def _client(self):
        """Return the imported remote package, or None when it is not installed."""
        raise NotImplementedError

    def _missing_package_message(self) -> str:
        """Error text used when the remote package is missing."""
        raise NotImplementedError

    def _init_run(self, log_dir: str, project: str, run_name: str) -> None:
        """Start the remote run."""
        raise NotImplementedError

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration."""
        self._update_config({"train_cfg": train_cfg})
        try:
            self._update_config({"env_cfg": env_cfg.to_dict()})  # type: ignore[union-attr]
        except Exception:
            self._update_config({"env_cfg": asdict(env_cfg)})  # type: ignore[arg-type]

    def _update_config(self, values: dict) -> None:
        """Merge one config dictionary into the remote run."""
        raise NotImplementedError

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to both TensorBoard and the remote run."""
        self.tensorboard_writer.add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        self._log_scalar(tag, scalar_value, global_step)

    def _log_scalar(self, tag: str, scalar_value: float, global_step: int | None) -> None:
        """Log one scalar to the remote run."""
        raise NotImplementedError

    def stop(self) -> None:
        """Close TensorBoard and finish the remote run."""
        self.tensorboard_writer.close()
        self._finish()

    def _finish(self) -> None:
        """Finish the remote run."""
        raise NotImplementedError

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint."""
        self._upload(model_path)

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file."""
        self._upload(path)

    def _upload(self, path: str) -> None:
        """Upload one file path."""
        raise NotImplementedError

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video once per filename."""
        if video.name in self.logged_videos:
            return
        self._upload_video(video, it)
        self.logged_videos.add(video.name)

    def _upload_video(self, video: pathlib.Path, it: int) -> None:
        """Upload one video file."""
        raise NotImplementedError


class WandbLogWriter(_RemoteLogWriter):
    """Summary writer for W&B."""

    service_name = "W&B"
    project_cfg_key = "wandb_project"

    def _client(self):
        return wandb

    def _missing_package_message(self) -> str:
        return "wandb package is required to log to Weights and Biases."

    def _init_run(self, log_dir: str, project: str, run_name: str) -> None:
        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None
        wandb.init(project=project, entity=entity, name=run_name, config={"log_dir": log_dir})

    def _update_config(self, values: dict) -> None:
        wandb.config.update(values)

    def _log_scalar(self, tag: str, scalar_value: float, global_step: int | None) -> None:
        wandb.log({tag: scalar_value}, step=global_step)

    def _finish(self) -> None:
        wandb.finish()

    def _upload(self, path: str) -> None:
        wandb.save(path, base_path=os.path.dirname(path))

    def _upload_video(self, video: pathlib.Path, it: int) -> None:
        wandb.log({"video": wandb.Video(str(video), format="mp4")}, step=it)


class SwanLabLogWriter(_RemoteLogWriter):
    """Summary writer for SwanLab."""

    service_name = "SwanLab"
    project_cfg_key = "swanlab_project"

    def _client(self):
        return swanlab

    def _missing_package_message(self) -> str:
        return "swanlab package is required to log to SwanLab."

    def _init_run(self, log_dir: str, project: str, run_name: str) -> None:
        try:
            workspace = os.environ["SWANLAB_WORKSPACE"]
        except KeyError:
            workspace = None
        swanlab.init(
            project=project,
            workspace=workspace,
            experiment_name=run_name,
            config={"log_dir": log_dir},
        )

    def _update_config(self, values: dict) -> None:
        swanlab.config.update(values)

    def _log_scalar(self, tag: str, scalar_value: float, global_step: int | None) -> None:
        swanlab.log({tag: float(scalar_value)}, step=global_step)

    def _finish(self) -> None:
        swanlab.finish()

    def _upload(self, path: str) -> None:
        swanlab.save(path, base_path=os.path.dirname(path), policy="now")

    def _upload_video(self, video: pathlib.Path, it: int) -> None:
        # SwanLab charts accept GIF. Other formats, including runner MP4 files, are stored as files.
        if video.suffix.lower() == ".gif":
            swanlab.log({"video": swanlab.Video(str(video), caption=video.name)}, step=it)
        else:
            swanlab.save(str(video), base_path=str(video.parent), policy="now")


class NeptuneLogWriter(LogWriter):
    """Summary writer for Neptune."""

    def __init__(
        self,
        log_dir: str,
        project_name: str | None = None,
        flush_secs: int = 10,
        cfg: dict | None = None,
    ) -> None:
        """Initialize a Neptune run for logging."""
        if neptune is None:
            raise ModuleNotFoundError("neptune-client is required to log to Neptune.")
        from torch.utils.tensorboard import SummaryWriter

        self.tensorboard_writer = SummaryWriter(log_dir, flush_secs=flush_secs)

        run_name = os.path.split(log_dir)[-1]
        project = project_name or (cfg or {}).get("neptune_project")
        if project is None:
            raise KeyError("Please specify Neptune project_name in logger config.")
        try:
            token = os.environ["NEPTUNE_API_TOKEN"]
        except KeyError:
            raise KeyError(
                "Neptune api token not found. Please run or add to ~/.bashrc: export NEPTUNE_API_TOKEN=YOUR_API_TOKEN"
            ) from None
        try:
            entity = os.environ["NEPTUNE_USERNAME"]
        except KeyError:
            raise KeyError(
                "Neptune username not found. Please run or add to ~/.bashrc: export NEPTUNE_USERNAME=YOUR_USERNAME"
            ) from None

        neptune_project = entity + "/" + project
        self.run = neptune.init_run(project=neptune_project, api_token=token)
        self.run["log_dir"].log(run_name)
        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration to Neptune."""
        self.run["train_cfg"] = train_cfg
        try:
            self.run["env_cfg"] = env_cfg.to_dict()  # type: ignore[union-attr]
        except Exception:
            self.run["env_cfg"] = asdict(env_cfg)  # type: ignore[arg-type]

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to both TensorBoard and Neptune."""
        self.tensorboard_writer.add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        self.run[self._map_path(tag)].log(scalar_value, step=global_step)

    def stop(self) -> None:
        """Finish the active Neptune run."""
        self.tensorboard_writer.close()
        self.run.stop()

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint artifact to Neptune."""
        self.run["model/saved_model_" + str(it)].upload(model_path)

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file artifact to Neptune."""
        name = path.rsplit("/", 1)[-1].split(".")[0]
        self.run["git_diff/" + name].upload(path)

    def _map_path(self, path: str) -> str:
        """Map metric names to Neptune-compatible keys."""
        if path in self.name_map:
            return self.name_map[path]
        return path
