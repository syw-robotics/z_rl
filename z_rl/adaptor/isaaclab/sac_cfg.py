"""IsaacLab configurations for Soft Actor-Critic."""

from __future__ import annotations

from isaaclab.utils import configclass

from .rl_cfg import ZRlBaseRunnerCfg, ZRlComposableModelCfg, ZRlMLPModelCfg


@configclass
class ZRlSquashedGaussianDistributionCfg(ZRlMLPModelCfg.DistributionCfg):
    class_name: str = "SquashedGaussianDistribution"
    init_std: float = 1.0
    action_low: float | list[float] | None = None
    action_high: float | list[float] | None = None


@configclass
class ZRlSacAlgorithmCfg:
    class_name: str = "SAC"
    replay_buffer_size: int = 1_000_000
    learning_starts: int = 1024
    batch_size: int = 256
    updates_per_iteration: int = 1
    n_steps: int = 1
    gamma: float = 0.998
    tau: float = 0.005
    policy_frequency: int = 2
    max_grad_norm: float = 1.0
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    alpha_learning_rate: float = 1e-3
    actor_optimizer: str = "adam"
    critic_optimizer: str = "adam"
    alpha: float = 0.05
    auto_alpha: bool = True
    target_entropy_scale: float = 1.0


@configclass
class ZRlSacRunnerCfg(ZRlBaseRunnerCfg):
    class_name: str = "OnPolicyRunner"
    num_steps_per_env: int = 1
    actor: ZRlComposableModelCfg = ZRlComposableModelCfg(
        hidden_dims=[256, 256, 256], activation="elu", distribution_cfg=ZRlSquashedGaussianDistributionCfg()
    )
    critic: ZRlComposableModelCfg = ZRlComposableModelCfg(
        class_name="SACCriticModel", hidden_dims=[256, 256, 256], activation="elu"
    )
    algorithm: ZRlSacAlgorithmCfg = ZRlSacAlgorithmCfg()
