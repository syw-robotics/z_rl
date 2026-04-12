# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .symmetry_cfg import ZRlSymmetryCfg

#########################
# Model configurations #
#########################


@configclass
class ZRlMLPModelCfg:
    """Configuration for the MLP model."""

    class_name: str = "MLPModel"
    """The model class name. Defaults to MLPModel."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the MLP network."""

    activation: str = MISSING
    """The activation function for the MLP network."""

    obs_normalization: bool = False
    """Whether to normalize the observation for the model. Defaults to False."""

    distribution_cfg: DistributionCfg | None = None
    """The configuration for the output distribution. Defaults to None, in which case no distribution is used."""

    @configclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = MISSING
        """The distribution class name."""

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = "GaussianDistribution"
        """The distribution class name. Default is GaussianDistribution."""

        init_std: float = MISSING
        """The initial standard deviation of the output distribution."""

        std_type: Literal["scalar", "log"] = "scalar"
        """The parameterization type of the output distribution's standard deviation. Default is scalar."""

    @configclass
    class HeteroscedasticGaussianDistributionCfg(GaussianDistributionCfg):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"
        """The distribution class name. Default is HeteroscedasticGaussianDistribution."""


@configclass
class ZRlRNNModelCfg(ZRlMLPModelCfg):
    """Configuration for RNN model."""

    class_name: str = "RNNModel"
    """The model class name. Defaults to RNNModel."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


@configclass
class ZRlMoEModelCfg:
    """Configuration for MoE model."""

    class_name: str = "MoEModel"
    """The model class name. Defaults to MoEModel."""

    activation: str = MISSING
    """The activation function for the expert and gate networks."""

    obs_normalization: bool = False
    """Whether to normalize the observation for the model. Defaults to False."""

    distribution_cfg: ZRlMLPModelCfg.DistributionCfg | None = None
    """The configuration for the output distribution. Defaults to None, in which case no distribution is used."""

    num_experts: int = 4
    """The number of experts in the MoE head."""

    expert_hidden_dims: list[int] = MISSING
    """The hidden dimensions of each expert network."""

    gate_hidden_dims: list[int] | None = None
    """The hidden dimensions of the gate network. If None, use a linear gate."""


@configclass
class ZRlCNNModelCfg(ZRlMLPModelCfg):
    """Configuration for CNN model."""

    class_name: str = "CNNModel"
    """The model class name. Defaults to CNNModel."""

    @configclass
    class CNNCfg:
        output_channels: tuple[int] | list[int] = MISSING
        """The number of output channels for each convolutional layer for the CNN."""

        kernel_size: int | tuple[int] | list[int] = MISSING
        """The kernel size for the CNN."""

        stride: int | tuple[int] | list[int] = 1
        """The stride for the CNN. Defaults to 1."""

        dilation: int | tuple[int] | list[int] = 1
        """The dilation for the CNN. Defaults to 1."""

        padding: Literal["none", "zeros", "reflect", "replicate", "circular"] = "none"
        """The padding for the CNN. Defaults to none."""

        norm: Literal["none", "batch", "layer"] | tuple[str] | list[str] = "none"
        """The normalization for the CNN. Defaults to none."""

        activation: str = MISSING
        """The activation function for the CNN."""

        max_pool: bool | tuple[bool] | list[bool] = False
        """Whether to use max pooling for the CNN. Defaults to False."""

        global_pool: Literal["none", "max", "avg"] = "none"
        """The global pooling for the CNN. Defaults to none."""

        flatten: bool = True
        """Whether to flatten the output of the CNN. Defaults to True."""

    @configclass
    class CNNProjectionCfg:
        hidden_dims: tuple[int] | list[int] = ()
        """Optional hidden dimensions of the post-CNN projection MLP."""

        output_dim: int = MISSING
        """Output dimension of the post-CNN projection MLP."""

        activation: str = "elu"
        """The activation function for the post-CNN projection MLP."""

        last_activation: str | None = None
        """Optional last activation for the post-CNN projection MLP."""

    cnn_cfg: CNNCfg = MISSING
    """The configuration for the CNN(s)."""

    cnn_projection_cfg: CNNProjectionCfg | None = None
    """Optional configuration for a projection MLP applied after flattened CNN features."""


############################
# Algorithm configurations #
############################


@configclass
class ZRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Defaults to PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use. Defaults to adam."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Defaults to False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    share_cnn_encoders: bool = False
    """Whether to share the CNN networks between actor and critic, in case CNNModels are used. Defaults to False."""

    symmetry_cfg: ZRlSymmetryCfg | None = None
    """The symmetry configuration. Defaults to None, in which case symmetry is not used."""


#########################
# Runner configurations #
#########################


@configclass
class ZRlBaseRunnerCfg:
    """Base configuration of the runner."""

    seed: int = 42
    """The seed for the experiment. Defaults to 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Defaults to cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation groups to observation sets.

    The keys of the dictionary are predefined observation sets used by the underlying algorithm
    and values are lists of observation groups provided by the environment.

    For instance, if the environment provides a dictionary of observations with groups "policy", "images",
    and "privileged", these can be mapped to algorithmic observation sets as follows:

    .. code-block:: python

        obs_groups = {
            "actor": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    This way, the actor will receive the "policy" and "images" observations, and the critic will
    receive the "policy" and "privileged" observations.

    For more details, please check ``vec_env.py`` in the z_rl library.
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`ZRlVecEnvWrapper` wrapper.
    """

    obs_group_concat_mode: Literal["term_major", "history_major"] = "term_major"
    """Concatenation layout for dict-based observation groups. Defaults to ``"term_major"``.

    - ``"term_major"``: Concatenate each term as-is in term order (Isaac Lab default layout).
    - ``"history_major"``: Reorder compatible vector history terms into a history-major flat layout.

    .. note::
        ``"history_major"`` only applies to compatible dict observation groups and requires
        ``concatenate_terms=False`` in Isaac Lab's observation group configuration.
        Incompatible groups will fall back to term-major behavior.
    """

    check_for_nan: bool = False
    """Whether to check for NaN values coming from the environment."""

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Defaults to empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Defaults to tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Defaults to "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Defaults to "isaaclab"."""

    resume: bool = False
    """Whether to resume a previous training. Defaults to False.

    This flag will be ignored for distillation.
    """

    load_run: str = ".*"
    """The run directory to load. Defaults to ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Defaults to ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@configclass
class ZRlOnPolicyRunnerCfg(ZRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Defaults to OnPolicyRunner."""

    actor: ZRlMLPModelCfg = MISSING
    """The actor configuration."""

    critic: ZRlMLPModelCfg = MISSING
    """The critic configuration."""

    algorithm: ZRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    check_nan: bool = False
    """Whether to check for NaN values during training. Defaults to False."""
