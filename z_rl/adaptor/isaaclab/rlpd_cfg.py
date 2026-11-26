"""IsaacLab presets for RLPD and the Active Adaptation BTX gated-BC variant."""

from isaaclab.utils import configclass

from .rl_cfg import ZRlComposableModelCfg
from .sac_cfg import ZRlSacAlgorithmCfg, ZRlSacRunnerCfg, ZRlSquashedGaussianDistributionCfg


@configclass
class ZRlRlpdAlgorithmCfg(ZRlSacAlgorithmCfg):
    class_name: str = "RLPD"
    prior_data: str | None = None
    prior_data_ratio: float = 0.5
    updates_per_iteration: int = 20
    policy_frequency: int | None = None
    gamma: float = 0.99
    alpha: float = 1.0
    target_entropy_scale: float = 0.5
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4


@configclass
class ZRlBtxRlpdAlgorithmCfg(ZRlRlpdAlgorithmCfg):
    class_name: str = "BTXRLPD"
    prior_data_ratio: float = 0.4
    updates_per_iteration: int = 4
    n_steps: int = 3
    alpha: float = 0.004
    bc_loss: str = "mse"
    bc_coef: float | None = None


@configclass
class ZRlRlpdCriticCfg(ZRlComposableModelCfg):
    class_name: str = "SACCriticModel"
    num_qs: int = 10
    num_min_qs: int = 2


@configclass
class ZRlRlpdRunnerCfg(ZRlSacRunnerCfg):
    algorithm: ZRlRlpdAlgorithmCfg = ZRlRlpdAlgorithmCfg()
    actor: ZRlComposableModelCfg = ZRlComposableModelCfg(
        hidden_dims=[256, 256], activation="relu", distribution_cfg=ZRlSquashedGaussianDistributionCfg()
    )
    critic: ZRlRlpdCriticCfg = ZRlRlpdCriticCfg(hidden_dims=[256, 256], activation="relu", layer_norm="pre_activation")


@configclass
class ZRlBtxRlpdRunnerCfg(ZRlSacRunnerCfg):
    algorithm: ZRlBtxRlpdAlgorithmCfg = ZRlBtxRlpdAlgorithmCfg()
