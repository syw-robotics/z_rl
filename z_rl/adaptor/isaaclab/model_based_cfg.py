"""IsaacLab configuration surface for the independent model-based runner."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from .rl_cfg import ZRlBaseRunnerCfg, ZRlMLPModelCfg


@configclass
class ZRlModelBasedRunnerCfg(ZRlBaseRunnerCfg):
    class_name: str = "ModelBasedRunner"
    actor: ZRlMLPModelCfg = MISSING
    critic: ZRlMLPModelCfg = MISSING
    algorithm: dict = {
        "class_name": "z_rl.algorithms.shac:SHAC",
        "optimizer": "adam",
        "horizon": 32,
        "gamma": 0.99,
        "lam": 0.95,
        "actor_learning_rate": 2e-4,
        "actor_weight_decay": 0.0,
        "critic_learning_rate": 2e-4,
        "critic_epochs": 5,
        "critic_minibatches": 4,
        "max_grad_norm": 1.0,
        "max_grad_value": 0.1,
        "compile": True,
    }
    world_model: dict = {
        "class_name": "z_rl.models.dynamics:LocalResidualDynamics",
        "hidden_dims": [2048, 2048, 2048],
        "activation": "swish",
        "layer_norm": "pre_activation",
        "input_normalization": True,
    }
    world_model_training: dict = {
        "optimizer": "adam",
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "unroll_length": 8,
        "batch_size": 2048,
        "updates_per_iteration": 10,
        "batch_updates": False,
        "checkpoint": None,
        "trainable": True,
    }
    replay_capacity_steps: int = 1000
    imagination_batch_size: int = 1024
    collect_steps_per_iteration: float = 2.5
    action_noise_std: float = 0.1
    warmup_steps: int = 64
    model_warmup_iterations: int = 500
    # Default for both updates. Set either override to capture only the desired update.
    capture_updates: bool = False
    capture_world_model_updates: bool | None = None
    capture_policy_updates: bool | None = None
    log_interval: int = 1
    defer_env_logs: bool = False
    actor_ema_decay: float = 0.0
    lr_decay_iterations: int = 250_000
    # Base runner compatibility; actual samples are counted from collection.
    num_steps_per_env: int = 1
