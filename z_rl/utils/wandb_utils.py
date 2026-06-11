# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import pathlib
from dataclasses import asdict

from z_rl.utils.log_writer import LogWriter

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class WandbLogWriter(LogWriter):
    """Summary writer for W&B."""

    def __init__(
        self,
        log_dir: str,
        project_name: str | None = None,
        flush_secs: int = 10,
        cfg: dict | None = None,
    ) -> None:
        """Initialize a W&B run for logging."""
        if wandb is None:
            raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.")
        from torch.utils.tensorboard import SummaryWriter

        self.tensorboard_writer = SummaryWriter(log_dir, flush_secs=flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        # Old callers pass cfg["wandb_project"]; new callers pass project_name directly.
        project = project_name or (cfg or {}).get("wandb_project")
        if project is None:
            raise KeyError("Please specify W&B project_name in logger config.")
        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config={"log_dir": log_dir},
            settings=wandb.Settings(start_method="thread"),
        )

        # Initialize set to keep track of logged videos
        self.logged_videos: set[str] = set()

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration to W&B."""
        wandb.config.update({"train_cfg": train_cfg})
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})  # type: ignore
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})  # type: ignore

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to both TensorBoard and W&B."""
        self.tensorboard_writer.add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        """Finish the active W&B run."""
        self.tensorboard_writer.close()
        wandb.finish()

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint artifact to W&B."""
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file artifact to W&B."""
        wandb.save(path, base_path=os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video artifact once per filename to W&B."""
        if video.name not in self.logged_videos:
            wandb.log({"video": wandb.Video(str(video), format="mp4")}, step=it)
            self.logged_videos.add(video.name)


WandbSummaryWriter = WandbLogWriter
