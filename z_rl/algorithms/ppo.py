# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict
from collections import defaultdict

from z_rl.env import VecEnv
from z_rl.extensions import Symmetry, resolve_symmetry_config
from z_rl.models import MLPModel
from z_rl.storage import RolloutStorage
from z_rl.utils import (
    compile_model,
    inject_obs_time_slice_map,
    reduce_gradients_in_buckets,
    resolve_callable,
    resolve_class,
    resolve_obs_groups,
)
from z_rl.utils.checkpoint import model_checkpoint_entries
from z_rl.utils.opt import build_optimizer


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_muon: bool = False,  # Muon for hidden 2-D weights; AdamW for biases/heads/std (see utils.opt)
        muon_adamw_weight_decay: float = 0.01,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        use_mixed_precision: bool = False,
        device: str = "cpu",
        # Symmetry parameters
        symmetry_augmentation: bool = False,
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        grad_reduce_bucket_mb: float = 25,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        self.grad_reduce_bucket_mb = grad_reduce_bucket_mb

        # Symmetry extension
        if symmetry_augmentation and symmetry_cfg is None:
            raise ValueError("symmetry_augmentation requires an environment-backed symmetry config.")
        if symmetry_cfg is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError(
                "Symmetry augmentation is not supported for recurrent policies: "
                "the hidden state has no explicit symmetry transform."
            )
        self.symmetry = (
            Symmetry(batch_is_augmented=symmetry_augmentation, **symmetry_cfg) if symmetry_cfg is not None else None
        )
        self._use_symmetry_augmentation = symmetry_augmentation
        self._use_mirror_loss = False
        self._mirror_loss_log_interval = None
        if self.symmetry is not None:
            self.mirror_loss_coef = self.symmetry.mirror_loss_coeff
            self._use_mirror_loss = self.symmetry.use_mirror_loss
            if self.symmetry.log_mirror_loss and not self._use_mirror_loss:
                self._mirror_loss_log_interval = self.symmetry.mirror_loss_log_interval

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        # Hidden 2-D weights use Muon when use_muon=True; biases, std, and input/output layers stay on AdamW.
        self.optimizer = build_optimizer(
            [self.actor, self.critic],
            learning_rate=learning_rate,
            optimizer=optimizer,
            use_muon=use_muon,
            muon_adamw_weight_decay=muon_adamw_weight_decay,
        )

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.use_mixed_precision = use_mixed_precision
        self.actor_forward_context = None
        self._update_iteration = 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.distribution.params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step."""
        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute the bootstrap value without advancing recurrent critic state.
        critic_hidden_state = self.critic.get_hidden_state()
        last_values = self.critic(obs).detach()
        self.critic.reset(hidden_state=critic_hidden_state)
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_losses = defaultdict(float)
        mirror_loss_metric = None
        log_mirror_loss = self._mirror_loss_log_interval is not None and (
            self._update_iteration % self._mirror_loss_log_interval == 0
        )

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for minibatch in generator:
            original_batch_size = minibatch.observations.batch_size[0]
            # Optionally use mixed precision for the forward pass and loss computation
            with torch.amp.autocast(  # type: ignore
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                opt_losses, non_opt_losses = self.compute_loss(minibatch)

            # Detached diagnostics are opt-in and evaluated on one mini-batch per selected PPO update.
            if log_mirror_loss:
                mirror_loss_metric = self.symmetry.compute_metric(  # type: ignore
                    self.actor, minibatch, original_batch_size
                )
                log_mirror_loss = False

            opt_loss = 0.0
            for k, v in opt_losses.items():
                opt_loss += getattr(self, k + "_coef", 1.0) * v
                mean_losses[k] = mean_losses[k] + v.detach()
            for k, v in non_opt_losses.items():
                mean_losses[k] = mean_losses[k] + v.detach()

            # Gradient step
            self.gradient_step(opt_loss)
            self.actor_forward_context = None

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in mean_losses.keys():
            mean_losses[k] = mean_losses[k] / num_updates
        if mirror_loss_metric is not None:
            mean_losses["mirror_loss_detach"] = mirror_loss_metric

        with torch.no_grad():
            values = self.storage.values.flatten()
            returns = self.storage.returns.flatten()
            # 1 means perfect value prediction, 0 matches a constant baseline, and negative values are worse.
            mean_losses["explained_variance"] = 1.0 - (returns - values).var(unbiased=False) / (
                returns.var(unbiased=False) + 1e-8
            )

        # Update the normalizers from the collected rollout after optimization
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

        # Clear the storage
        self.storage.clear()
        self._update_iteration += 1

        return mean_losses

    def compute_loss(self, minibatch: RolloutStorage.Batch) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        original_batch_size = minibatch.observations.batch_size[0]

        # Check if we should normalize advantages per mini batch
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                minibatch.advantages = (minibatch.advantages - minibatch.advantages.mean()) / (
                    minibatch.advantages.std() + 1e-8
                )  # type: ignore

        # Perform symmetric augmentation
        if self._use_symmetry_augmentation:
            self.symmetry.augment_batch(minibatch, original_batch_size)

        # Recompute actions log prob and entropy for current batch of transitions
        # Note: We need to do this because we updated the policy with the new parameters
        self.forward_actor_for_update(minibatch)
        actions_log_prob = self.actor.get_output_log_prob(minibatch.actions)  # type: ignore
        values = self.critic(minibatch.observations, masks=minibatch.masks, hidden_state=minibatch.hidden_states[1])
        # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
        distribution_params = tuple(p[:original_batch_size] for p in self.actor.distribution.params)
        entropy = self.actor.output_entropy[:original_batch_size]

        # Compute KL divergence for diagnostics and adaptive learning-rate scheduling
        with torch.inference_mode():
            kl = self.actor.get_kl_divergence(minibatch.old_distribution_params, distribution_params)  # type: ignore
            kl_mean = torch.mean(kl)

            # Reduce the KL divergence across all GPUs
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size

            if self.desired_kl is not None and self.schedule == "adaptive":
                # Update the learning rate only on the main process
                if self.gpu_global_rank == 0:
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                # Update the learning rate for all GPUs
                if self.is_multi_gpu:
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()

                # Update the learning rate for all parameter groups
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob - torch.squeeze(minibatch.old_actions_log_prob))  # type: ignore
        surrogate = -torch.squeeze(minibatch.advantages) * ratio  # type: ignore
        surrogate_clipped = -torch.squeeze(minibatch.advantages) * torch.clamp(  # type: ignore
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = minibatch.values + (values - minibatch.values).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - minibatch.returns).pow(2)
            value_losses_clipped = (value_clipped - minibatch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (minibatch.returns - values).pow(2).mean()

        # pack the losses and metrics dicts
        opt_losses = dict(
            surrogate_loss=surrogate_loss,
            value_loss=value_loss,
            entropy=-entropy.mean(),
        )
        non_opt_losses = dict(
            clip_fraction=((ratio - 1.0).abs() > self.clip_param).float().mean(),
            kl=kl_mean,
        )

        # Symmetry loss
        if self._use_mirror_loss:
            symmetry_loss = self.symmetry.compute_loss(self.actor, minibatch, original_batch_size)
            opt_losses["mirror_loss"] = symmetry_loss

        return opt_losses, non_opt_losses

    def forward_actor_for_update(self, minibatch: RolloutStorage.Batch) -> None:
        """Run the actor forward pass used by update-time loss computation.

        Subclasses may override this to expose differentiable forward context for additional loss specs while keeping
        the base PPO loss computation unchanged.
        """
        self.actor_forward_context = None
        _, self.actor_forward_context = self.actor.forward_with_context(
            minibatch.observations,
            masks=minibatch.masks,
            hidden_state=minibatch.hidden_states[0],
            stochastic_output=True,
        )

    def gradient_step(self, loss: torch.Tensor):
        # Compute the gradients for PPO
        self.optimizer.zero_grad()
        loss.backward()

        # Collect gradients from all GPUs
        if self.is_multi_gpu:
            self.reduce_parameters()

        # Apply the gradients for PPO
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            **model_checkpoint_entries("actor", self._raw_actor),
            **model_checkpoint_entries("critic", self._raw_critic),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic for training while keeping raw modules for checkpoints/export."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    @classmethod
    def _build_algorithm_extra_kwargs(cls, env: VecEnv, algorithm_cfg: dict) -> dict:
        """Build subclass-specific keyword arguments for algorithm construction."""
        del env, algorithm_cfg
        return {}

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables without mutating the original config (keeps class names for logging).
        alg_class: type[PPO] = cls
        alg_cfg = copy.deepcopy(cfg["algorithm"])
        class_name = alg_cfg.pop("class_name", None)
        if cls is PPO and class_name is not None:
            alg_class = resolve_callable(class_name)  # type: ignore
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve symmetry config if used
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        # Inject wrapper-provided time-slice metadata for models that can consume it.
        inject_obs_time_slice_map(actor_cfg, actor_class, env)
        inject_obs_time_slice_map(critic_cfg, critic_class, env)

        # Pop init_weights configs before creating models (they are not model __init__ args)
        actor_init_weights = actor_cfg.pop("init_weights", None)
        actor_cnn_init_weights = actor_cfg.pop("cnn_init_weights", None)
        critic_init_weights = critic_cfg.pop("init_weights", None)
        critic_cnn_init_weights = critic_cfg.pop("cnn_init_weights", None)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")
        if alg_cfg.pop("share_cnn_encoders", None):
            raise ValueError(
                "share_cnn_encoders is no longer supported; actor and critic each build their own CNN encoder."
            )
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")

        # Initialize weights if configured
        if actor_init_weights is not None:
            actor.head.init_weights(actor_init_weights)
            print("-" * 80)
            print(f"Actor Head uses orthogonal init: {actor_init_weights}")
        if critic_init_weights is not None:
            critic.head.init_weights(critic_init_weights)
            print(f"Critic Head uses orthogonal init: {critic_init_weights}")
        # Initialize CNN weights if configured
        if actor_cnn_init_weights:
            actor.latent_adapter.cnn.init_cnn_weights()
            print("Actor CNNs use kaiming init")
        if critic_cnn_init_weights:
            critic.latent_adapter.cnn.init_cnn_weights()
            print("Critic CNNs use kaiming init")
        print("-" * 80)

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        extra_kwargs = alg_class._build_algorithm_extra_kwargs(env, alg_cfg)

        # Initialize the algorithm
        alg: PPO = alg_class(
            actor,
            critic,
            storage,
            device=device,
            **alg_cfg,
            **extra_kwargs,
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        for model in (self._raw_actor, self._raw_critic):
            for tensor in model.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        params = chain(self.actor.parameters(), self.critic.parameters())
        reduce_gradients_in_buckets(params, self.gpu_world_size, self.grad_reduce_bucket_mb)
