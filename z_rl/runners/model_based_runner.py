# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect real transitions, fit dynamics, then optimize actor and critic in imagination.

The environment owns the task contract: num_joints, context_dim, num_frames,
max_episode_length, observations, add_noise, and evaluate. The wrapper stores flat
pre-reset transitions in extras["model_based_transition"].
"""

from __future__ import annotations

import math
import os
import time
import torch
from copy import deepcopy
from tensordict import TensorDict

from z_rl.env import VecEnv
from z_rl.storage.sequence_replay import SequenceReplayBuffer
from z_rl.utils import resolve_class
from z_rl.utils.cuda_graph import (
    GraphedUpdate,
    clip_grad_norm,
    graph_config,
    graph_state,
    make_capturable,
    set_learning_rate,
)
from z_rl.utils.logger import Logger
from z_rl.utils.onnx_export import export_to_onnx
from z_rl.utils.opt import build_optimizer


class ModelBasedRunner:
    """Collect real transitions, fit a world model, then update the policy in imagination."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the actor, and the training stack when a log directory is given."""
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.current_learning_iteration = 0
        self.sim_steps = 0
        self._collect_fraction = 0.0
        self._unlogged_samples = 0
        self.checkpoint_infos: dict = {}

        # Playing and exporting only need the actor.
        obs = self.env.get_observations().to(self.device)
        self.actor = self._build_model(obs, train_cfg["actor"], "actor", self.env.num_actions)
        self.training = log_dir is not None
        self.actor_ema = None
        self._actor_ema_updates = 0
        self.logger = Logger(
            log_dir=log_dir,
            cfg=train_cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=False,
            gpu_world_size=1,
            gpu_global_rank=0,
            device=self.device,
        )

        if not self.training:
            return
        self._build_training_stack(obs)
        self._configure_capture()

    def _build_model(self, obs: TensorDict, model_cfg: dict, role: str, output_dim: int) -> torch.nn.Module:
        """Build an actor or critic from a copied model config."""
        model_class, model_kwargs = resolve_class(model_cfg)
        if role == "actor" and hasattr(self.env, "obs_format"):
            model_kwargs.setdefault("obs_format", self.env.obs_format)
        init_weights = model_kwargs.pop("init_weights", None)
        if role == "actor" and model_kwargs.get("distribution_cfg") is not None:
            raise ValueError("ModelBasedRunner requires a deterministic actor (distribution_cfg=None)")
        model = model_class(obs, self.cfg["obs_groups"], role, output_dim, **model_kwargs).to(self.device)
        if init_weights is not None:
            model.head.init_weights(init_weights)
        return model

    def _build_training_stack(self, obs: TensorDict) -> None:
        """Build the critic, world model, replay buffer, and imagination algorithm."""
        self.actor_ema_decay = self.cfg.get("actor_ema_decay", 0.0)
        if self.actor_ema_decay:
            self.actor_ema = deepcopy(self.actor).requires_grad_(False).eval()
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("ModelBasedRunner currently supports one training process")

        # The critic also receives the task context.
        self.task = self.env.unwrapped.model_based_task.to(self.device)
        critic_obs = obs.set("context", self.env.unwrapped.model_based_context.to(self.device))
        self.critic = self._build_model(critic_obs, self.cfg["critic"], "critic", 1)
        self._build_world_model()

        self.replay = SequenceReplayBuffer(self.cfg["replay_capacity_steps"], self.env.num_envs, self.device)
        if self.replay.capacity < self.wm_cfg["unroll_length"]:
            raise ValueError("Replay capacity must cover the world-model unroll length")
        if self.cfg["collect_steps_per_iteration"] <= 0:
            raise ValueError("collect_steps_per_iteration must be positive")

        algorithm_class, algorithm_cfg = resolve_class(self.cfg["algorithm"])
        self.alg = algorithm_class(
            self.actor,
            self.critic,
            self.world_model,
            self.task,
            clip_actions=self.cfg.get("clip_actions"),
            **algorithm_cfg,
        )

        # A specific flag overrides capture_updates. Frozen dynamics are never captured.
        self.batch_wm_updates = self.wm_cfg.get("batch_updates", False)
        capture_updates = self.cfg.get("capture_updates", False)
        capture_world_model = self.cfg.get("capture_world_model_updates")
        capture_policy = self.cfg.get("capture_policy_updates")
        world_model_capture = capture_updates if capture_world_model is None else capture_world_model
        self.capture_world_model_updates = self.wm_trainable and world_model_capture
        self.capture_policy_updates = capture_updates if capture_policy is None else capture_policy

    def _build_world_model(self) -> None:
        """Build the dynamics model and the optimizer used to train or restore it."""
        model_class, model_cfg = resolve_class(self.cfg["world_model"])
        self.world_model = model_class(
            num_joints=self.task.num_joints,
            action_dim=self.env.num_actions,
            context_dim=self.task.context_dim,
            **model_cfg,
        ).to(self.device)
        self.wm_cfg = self.cfg["world_model_training"]
        self.wm_trainable = self.wm_cfg["trainable"]
        checkpoint_path = self.wm_cfg.get("checkpoint")
        if not self.wm_trainable and not checkpoint_path and not self.cfg.get("resume"):
            raise ValueError("A frozen world model needs a pretrained checkpoint or a training resume")
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self._check_task_metadata(checkpoint)
            self.world_model.load_state_dict(checkpoint["world_model_state_dict"])

        # Create optimizer state for every parameter, including a model that is then frozen.
        self.world_model.requires_grad_(True)
        self.wm_optimizer = build_optimizer(
            [self.world_model],
            learning_rate=self.wm_cfg["learning_rate"],
            optimizer=self.wm_cfg.get("optimizer", "adam"),
            use_muon=self.wm_cfg.get("use_muon", False),
            muon_adamw_weight_decay=0.0,
            weight_decay=self.wm_cfg.get("weight_decay", 0.0),
        )
        self.world_model.requires_grad_(self.wm_trainable)
        self.world_model.train(self.wm_trainable)

    def _configure_capture(self) -> None:
        """Wrap the selected updates so CUDA can replay them."""
        capturing = self.capture_world_model_updates or self.capture_policy_updates
        if capturing and torch.device(self.device).type != "cuda":
            raise ValueError("capture_updates requires a CUDA training device")
        if self.capture_policy_updates:
            validate_capture = getattr(self.alg, "validate_capture", None)
            if validate_capture is not None:
                validate_capture()

        # Optimizers must be capturable before a graph records them. The callbacks read the current objects,
        # because loading a checkpoint can replace them.
        self._prepare_capture_optimizers()
        if self.capture_world_model_updates:
            update = self._update_world_models if self.batch_wm_updates else self._update_world_model
            graphed = GraphedUpdate(
                update,
                state=lambda: graph_state([self.world_model], [self.wm_optimizer]),
                config=lambda: graph_config([self.world_model], [self.wm_optimizer]),
            )
            if self.batch_wm_updates:
                self._update_world_models = graphed
            else:
                self._update_world_model = graphed
        if self.capture_policy_updates:
            self._update_policy = GraphedUpdate(
                self._update_policy,
                state=lambda: graph_state(
                    [self.actor, self.critic, self.world_model, self.task],
                    [self.alg.actor_optimizer, self.alg.critic_optimizer],
                ),
                config=lambda: graph_config(
                    [self.alg, self.actor, self.critic, self.world_model, self.task],
                    [self.alg.actor_optimizer, self.alg.critic_optimizer],
                ),
            )

    def _prepare_capture_optimizers(self) -> None:
        """Make the active optimizers safe to record, before training and after loading."""
        if self.capture_world_model_updates:
            self.wm_optimizer = make_capturable(self.wm_optimizer)
        if self.capture_policy_updates:
            self.alg.actor_optimizer = make_capturable(self.alg.actor_optimizer)
            self.alg.critic_optimizer = make_capturable(self.alg.critic_optimizer)

    @torch.no_grad()
    def collect(self, steps: int) -> None:
        """Step the real environment and store its flat transitions."""
        for _ in range(steps):
            obs = self.env.get_observations().to(self.device)
            obs = obs.set("context", self.env.unwrapped.model_based_context.to(self.device))
            self.actor.update_normalization(obs)
            self.critic.update_normalization(obs)

            # Exploration noise stays outside the deterministic actor.
            action = self.actor(obs)
            action = action + torch.randn_like(action) * self.cfg["action_noise_std"]
            _, reward, done, extras = self.env.step(action.to(self.env.device))

            transition = {key: value.to(self.device) for key, value in extras["model_based_transition"].items()}
            self.replay.add(transition)
            if self.wm_trainable:
                self.world_model.update_normalization(transition["state"], transition["action"], transition["context"])
            self.logger.process_env_step(reward.to(self.device), done.to(self.device), extras)
            self.sim_steps += self.env.num_envs
            self._unlogged_samples += self.env.num_envs

    def _schedule(self, iteration: int) -> float:
        """Decay the world-model, actor, and critic learning rates together."""
        duration = self.cfg["lr_decay_iterations"]
        fraction = min(iteration / duration, 1.0) if duration else 0.0
        # Cosine decay from 1 to 0.03.
        factor = 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * fraction))
        schedules = (
            (self.wm_optimizer, self.wm_cfg["learning_rate"]),
            (self.alg.actor_optimizer, self.cfg["algorithm"]["actor_learning_rate"]),
            (self.alg.critic_optimizer, self.cfg["algorithm"]["critic_learning_rate"]),
        )
        for optimizer, base_learning_rate in schedules:
            for group in optimizer.param_groups:
                set_learning_rate(group, base_learning_rate * factor)
        self.alg.learning_rate = self.cfg["algorithm"]["actor_learning_rate"] * factor
        return factor

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Collect transitions, fit the world model, and update the actor and critic.

        Args:
            num_learning_iterations: Number of iterations to run.
            init_at_random_ep_len: Accepted by the shared runner interface. Episode length stays with the task.
        """
        self.logger.init_logging_writer()
        self.actor.train()
        self.critic.train()

        # Fill the replay buffer before the first dynamics update.
        required = max(self.cfg["warmup_steps"], self.wm_cfg["unroll_length"])
        start = time.perf_counter()
        self.collect(max(required - self.replay.size, 0))
        warmup_time = time.perf_counter() - start

        start_iteration = self.current_learning_iteration
        total_iterations = start_iteration + num_learning_iterations
        pending_collect = 0.0
        pending_learn = 0.0
        for iteration in range(start_iteration, total_iterations):
            self._schedule(iteration)

            # Collect real transitions. The first logged iteration includes replay warmup.
            start = time.perf_counter()
            self._collect_fraction += self.cfg["collect_steps_per_iteration"]
            steps = int(self._collect_fraction)
            self._collect_fraction -= steps
            self.collect(steps)
            collect_time = time.perf_counter() - start + warmup_time
            warmup_time = 0.0

            # Fit the dynamics, then update the policy in imagination after the model warmup.
            start = time.perf_counter()
            metrics: dict = {}
            if self.wm_trainable:
                metrics.update(self._fit_world_model())
            if iteration >= self.cfg["model_warmup_iterations"] or not self.wm_trainable:
                start_state = self.replay.sample_starts(self.cfg["imagination_batch_size"])
                metrics.update(self._update_policy(start_state))
                if self.actor_ema is not None:
                    self._update_actor_ema()
            learn_time = time.perf_counter() - start

            finished_iteration = iteration + 1
            self.current_learning_iteration = finished_iteration
            pending_collect += collect_time
            pending_learn += learn_time
            log_interval = self.cfg.get("log_interval", 1)
            should_log = finished_iteration % log_interval == 0 or finished_iteration == total_iterations
            if should_log:
                self.logger.log(
                    it=iteration,
                    start_it=start_iteration,
                    total_it=total_iterations,
                    collect_time=pending_collect,
                    learn_time=pending_learn,
                    loss_dict=metrics,
                    learning_rate=self.alg.learning_rate,
                    collection_size=self._unlogged_samples,
                )
                self._unlogged_samples = 0
                pending_collect = 0.0
                pending_learn = 0.0

            # Save a checkpoint.
            if self.current_learning_iteration % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

        # Save the final model after training.
        self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.logger.stop_logging_writer()

    def _fit_world_model(self) -> dict:
        """Sample replay windows and take the configured number of dynamics steps."""
        fields = getattr(self.world_model, "replay_fields", None)
        batch_size = self.wm_cfg["batch_size"]
        unroll_length = self.wm_cfg["unroll_length"]
        updates = self.wm_cfg["updates_per_iteration"]
        if self.batch_wm_updates:
            windows = [self.replay.sample_windows(batch_size, unroll_length, fields=fields) for _ in range(updates)]
            batches = {key: torch.stack([window[key] for window in windows]) for key in windows[0]}
            return self._update_world_models(batches)

        metrics = {}
        for _ in range(updates):
            window = self.replay.sample_windows(batch_size, unroll_length, fields=fields)
            metrics.update(self._update_world_model(window))
        return metrics

    def _update_world_models(self, batches: dict[str, torch.Tensor]) -> dict:
        """Step the world model once per stacked window. CUDA capture replays this whole loop."""
        count = next(iter(batches.values())).shape[0]
        for index in range(count):
            window = {key: value[index] for key, value in batches.items()}
            metrics = self._update_world_model(window)
        return metrics

    def _update_world_model(self, window: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Take one world-model optimizer step."""
        loss = self.world_model.loss(window)
        self.wm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm(self.world_model.parameters(), 1.0)
        self.wm_optimizer.step()
        return {"world_model": loss.detach()}

    def _update_policy(self, start: dict) -> dict:
        """Update the actor and critic from one batch of imagined rollouts."""
        return self.alg.update(start)

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state, then publish the checkpoint."""
        checkpoint = {
            "format_version": 1,
            "actor_state_dict": self._inference_actor().state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "world_model_state_dict": self.world_model.state_dict(),
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
            "world_model_optimizer_state_dict": self.wm_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "sim_steps": self.sim_steps,
            "collect_fraction": self._collect_fraction,
            "cfg": self.cfg,
            "task_metadata": getattr(self.task, "metadata", {}),
            "infos": {**self.checkpoint_infos, **(infos or {})},
        }
        if self.actor_ema is not None:
            checkpoint["actor_training_state_dict"] = self.actor.state_dict()
            checkpoint["actor_ema_updates"] = self._actor_ema_updates
        temporary = f"{path}.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a checkpoint.

        Args:
            path: Path to load the model from.
            load_cfg: Accepted by the shared runner interface. The checkpoint restores the full training state.
            strict: Whether state_dict loading should be strict.
            map_location: Device mapping for loading the model.
        """
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Expected a ModelBasedRunner checkpoint")
        if self.training:
            self._check_task_metadata(checkpoint)
            for update in (self._update_world_model, self._update_world_models, self._update_policy):
                if isinstance(update, GraphedUpdate):
                    update.invalidate()

        # The saved actor is the exported policy. Training resumes the live weights when both are stored.
        if self.training:
            actor_state = checkpoint.get("actor_training_state_dict", checkpoint["actor_state_dict"])
        else:
            actor_state = checkpoint["actor_state_dict"]
        self.actor.load_state_dict(actor_state, strict=strict)
        if self.training:
            if self.actor_ema is not None:
                self.actor_ema.load_state_dict(checkpoint["actor_state_dict"], strict=strict)
                self._actor_ema_updates = checkpoint.get("actor_ema_updates", 1)
            self.critic.load_state_dict(checkpoint["critic_state_dict"], strict=strict)
            self.world_model.load_state_dict(checkpoint["world_model_state_dict"], strict=strict)
            self.alg.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.alg.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.wm_optimizer.load_state_dict(checkpoint["world_model_optimizer_state_dict"])
            self._prepare_capture_optimizers()

        self.current_learning_iteration = checkpoint["iter"]
        self.sim_steps = checkpoint.get("sim_steps", 0)
        self.logger.tot_timesteps = self.sim_steps
        self._collect_fraction = checkpoint.get("collect_fraction", 0.0)
        self.checkpoint_infos = checkpoint.get("infos") or {}
        return self.checkpoint_infos

    def _check_task_metadata(self, checkpoint: dict) -> None:
        """Reject a dynamics checkpoint whose task definition does not match this environment."""
        if checkpoint.get("task_metadata", {}) != getattr(self.task, "metadata", {}):
            raise ValueError("World-model task metadata differs (asset, state/context convention or control period)")

    def _inference_actor(self) -> torch.nn.Module:
        """Return the exported actor. An EMA replaces the live actor when it is enabled."""
        if self.actor_ema is not None:
            return self.actor_ema
        return self.actor

    def get_inference_policy(self, device: str | None = None) -> torch.nn.Module:
        """Return the exported policy on the requested device."""
        return self._inference_actor().eval().to(device or self.device)

    @torch.no_grad()
    def _update_actor_ema(self) -> None:
        """Move the exported actor toward the live actor."""
        if self._actor_ema_updates == 0:
            # Include the normalization statistics gathered while the world model was warming up.
            self.actor_ema.load_state_dict(self.actor.state_dict())
            self._actor_ema_updates = 1
            return

        averaged = list(self.actor_ema.parameters())
        current = list(self.actor.parameters())
        for averaged_buffer, live_buffer in zip(self.actor_ema.buffers(), self.actor.buffers()):
            if averaged_buffer.is_floating_point():
                averaged.append(averaged_buffer)
                current.append(live_buffer)
            else:
                # Normalization sample counts are integers and are copied rather than averaged.
                averaged_buffer.copy_(live_buffer)
        torch._foreach_lerp_(averaged, current, 1 - self.actor_ema_decay)
        self._actor_ema_updates += 1

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the inference actor to an ONNX file."""
        model = self._inference_actor().as_onnx(verbose=verbose).cpu().eval()
        os.makedirs(path, exist_ok=True)
        export_to_onnx(
            model,
            model.get_dummy_inputs(),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=model.input_names,
            output_names=model.output_names,
        )

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the inference actor to a traced TorchScript file."""
        model = self._inference_actor().as_onnx(verbose=False).cpu().eval()
        os.makedirs(path, exist_ok=True)
        torch.jit.trace(model, model.get_dummy_inputs()).save(os.path.join(path, filename))
