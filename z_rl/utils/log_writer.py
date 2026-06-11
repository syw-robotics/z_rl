# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod


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
