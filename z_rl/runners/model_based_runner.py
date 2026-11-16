"""Collect real transitions, fit dynamics, then optimize actor/critic in imagination.

Task contract (owned by the environment): num_joints, context_dim, num_frames,
max_episode_length; observations, add_noise, evaluate. The wrapper passes flat
pre-reset transitions under extras['model_based_transition'].
"""

import math
import os
import time
from copy import deepcopy

import torch
from z_rl.storage.sequence_replay import SequenceReplayBuffer
from z_rl.utils import resolve_class
from z_rl.utils.logger import Logger
from z_rl.utils.onnx_export import export_to_onnx
from z_rl.utils.opt import build_optimizer
from z_rl.utils.cuda_graph import GraphedUpdate, graph_state, graph_config, clip_grad_norm, make_capturable, set_learning_rate


class ModelBasedRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.cfg, self.device = env, train_cfg, device
        self.current_learning_iteration, self.sim_steps, self._collect_fraction = 0, 0, 0.0
        self._unlogged_samples = 0
        self.checkpoint_infos = {}
        obs = env.get_observations().to(device)
        actor_class, actor_cfg = resolve_class(train_cfg["actor"])
        if hasattr(env, "obs_format"):
            actor_cfg.setdefault("obs_format", env.obs_format)
        actor_init = actor_cfg.pop("init_weights", None)
        if actor_cfg.get("distribution_cfg") is not None:
            raise ValueError("ModelBasedRunner requires a deterministic actor (distribution_cfg=None)")
        self.actor = actor_class(obs, train_cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        if actor_init is not None:
            self.actor.head.init_weights(actor_init)
        self.training = log_dir is not None
        self.actor_ema = None
        self._actor_ema_updates = 0
        self.logger = Logger(log_dir, train_cfg, env.cfg, env.num_envs, False, 1, 0, device)
        if not self.training:
            return  # Playing/exporting builds only the actor.
        self.actor_ema_decay = train_cfg.get("actor_ema_decay", 0.0)
        if self.actor_ema_decay:
            self.actor_ema = deepcopy(self.actor).requires_grad_(False).eval()
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("ModelBasedRunner currently supports one training process")
        self.task = env.unwrapped.model_based_task.to(device)
        obs = obs.set("context", env.unwrapped.model_based_context.to(device))
        critic_class, critic_cfg = resolve_class(train_cfg["critic"])
        critic_init = critic_cfg.pop("init_weights", None)
        self.critic = critic_class(obs, train_cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        if critic_init is not None:
            self.critic.head.init_weights(critic_init)
        wm_class, wm_cfg = resolve_class(train_cfg["world_model"])
        self.world_model = wm_class(
            num_joints=self.task.num_joints, action_dim=env.num_actions, context_dim=self.task.context_dim, **wm_cfg
        ).to(device)
        self.wm_cfg = train_cfg["world_model_training"]
        self.wm_trainable = self.wm_cfg["trainable"]
        if not self.wm_trainable and self.wm_cfg.get("checkpoint") is None and not train_cfg.get("resume"):
            raise ValueError("A frozen world model needs a pretrained checkpoint or a training resume")
        if self.wm_cfg.get("checkpoint"):
            ckpt = torch.load(self.wm_cfg["checkpoint"], map_location=device, weights_only=False)
            self._check_task_metadata(ckpt)
            self.world_model.load_state_dict(ckpt["world_model_state_dict"])
        self.world_model.requires_grad_(self.wm_trainable)
        self.world_model.train(self.wm_trainable)
        self.wm_optimizer = build_optimizer(
            [self.world_model], learning_rate=self.wm_cfg["learning_rate"],
            use_muon=self.wm_cfg.get("use_muon", False), muon_adamw_weight_decay=0.0,
        )
        for group in self.wm_optimizer.param_groups:
            group["weight_decay"] = 0.0
        self.replay = SequenceReplayBuffer(train_cfg["replay_capacity_steps"], env.num_envs, device)
        if self.replay.capacity < self.wm_cfg["unroll_length"]:
            raise ValueError("Replay capacity must cover the world-model unroll length")
        if train_cfg["collect_steps_per_iteration"] <= 0:
            raise ValueError("collect_steps_per_iteration must be positive")
        algorithm_class, algorithm_cfg = resolve_class(train_cfg["algorithm"])
        self.alg = algorithm_class(
            self.actor,
            self.critic,
            self.world_model,
            self.task,
            clip_actions=train_cfg.get("clip_actions"),
            **algorithm_cfg,
        )

        self.batch_wm_updates = self.wm_cfg.get("batch_updates", False)
        if train_cfg.get("capture_updates", False):
            if torch.device(device).type != "cuda":
                raise ValueError("capture_updates requires a CUDA training device")
            validate_capture = getattr(self.alg, "validate_capture", None)
            if validate_capture is not None:
                validate_capture()
            # Sampling stays outside capture; each call copies fresh tensors into stable buffers.
            self.wm_optimizer = make_capturable(self.wm_optimizer)
            self.alg.actor_optimizer = make_capturable(self.alg.actor_optimizer)
            self.alg.critic_optimizer = make_capturable(self.alg.critic_optimizer)
            wm_state = lambda: graph_state([self.world_model], [self.wm_optimizer])
            policy_state = lambda: graph_state(
                [self.actor, self.critic, self.world_model, self.task],
                [self.alg.actor_optimizer, self.alg.critic_optimizer],
            )
            wm_config = lambda: graph_config([self.world_model], [self.wm_optimizer])
            policy_config = lambda: graph_config(
                [self.alg, self.actor, self.critic, self.world_model, self.task],
                [self.alg.actor_optimizer, self.alg.critic_optimizer],
            )
            if self.batch_wm_updates:
                self._update_world_models = GraphedUpdate(self._update_world_models, wm_state, wm_config)
            else:
                self._update_world_model = GraphedUpdate(self._update_world_model, wm_state, wm_config)
            self._update_policy = GraphedUpdate(self._update_policy, policy_state, policy_config)

    @torch.no_grad()
    def collect(self, steps):
        for _ in range(steps):
            obs = self.env.get_observations().to(self.device)
            obs = obs.set("context", self.env.unwrapped.model_based_context.to(self.device))
            self.actor.update_normalization(obs)
            self.critic.update_normalization(obs)
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

    def _schedule(self, iteration):
        duration = self.cfg["lr_decay_iterations"]
        fraction = min(iteration / duration, 1.0) if duration else 0.0
        factor = 0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * fraction))
        for optimizer, base in (
            (self.wm_optimizer, self.wm_cfg["learning_rate"]),
            (self.alg.actor_optimizer, self.cfg["algorithm"]["actor_learning_rate"]),
            (self.alg.critic_optimizer, self.cfg["algorithm"]["critic_learning_rate"]),
        ):
            for group in optimizer.param_groups:
                set_learning_rate(group, base * factor)
        self.alg.learning_rate = self.cfg["algorithm"]["actor_learning_rate"] * factor
        return factor

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self.logger.init_logging_writer()
        self.actor.train()
        self.critic.train()
        required = max(self.cfg["warmup_steps"], self.wm_cfg["unroll_length"])
        start_time = time.perf_counter()
        self.collect(max(required - self.replay.size, 0))
        warmup_time = time.perf_counter() - start_time
        first = self.current_learning_iteration
        pending_collect, pending_learn = 0.0, 0.0
        for iteration in range(first, first + num_learning_iterations):
            self._schedule(iteration)
            begin = time.perf_counter()
            self._collect_fraction += self.cfg["collect_steps_per_iteration"]
            steps = int(self._collect_fraction)
            self._collect_fraction -= steps
            self.collect(steps)
            collect_time = time.perf_counter() - begin + warmup_time
            warmup_time = 0.0
            begin = time.perf_counter()
            metrics = {}
            if self.wm_trainable:
                fields = getattr(self.world_model, "replay_fields", None)
                if self.batch_wm_updates:
                    windows = [self.replay.sample_windows(
                        self.wm_cfg["batch_size"], self.wm_cfg["unroll_length"], fields=fields
                    ) for _ in range(self.wm_cfg["updates_per_iteration"])]
                    batches = {key: torch.stack([window[key] for window in windows]) for key in windows[0]}
                    metrics.update(self._update_world_models(batches))
                else:
                    for _ in range(self.wm_cfg["updates_per_iteration"]):
                        window = self.replay.sample_windows(
                            self.wm_cfg["batch_size"], self.wm_cfg["unroll_length"], fields=fields
                        )
                        metrics.update(self._update_world_model(window))
            if iteration >= self.cfg["model_warmup_iterations"] or not self.wm_trainable:
                metrics.update(self._update_policy(self.replay.sample_starts(self.cfg["imagination_batch_size"])))
                if self.actor_ema is not None:
                    self._update_actor_ema()
            self._after_update(iteration + 1)
            learn_time = time.perf_counter() - begin
            self.current_learning_iteration = iteration + 1
            pending_collect += collect_time
            pending_learn += learn_time
            if (iteration + 1) % self.cfg.get("log_interval", 1) == 0 or iteration + 1 == first + num_learning_iterations:
                self.logger.log(
                    iteration,
                    first,
                    first + num_learning_iterations,
                    pending_collect,
                    pending_learn,
                    metrics,
                    self.alg.learning_rate,
                    collection_size=self._unlogged_samples,
                )
                self._unlogged_samples = 0
                pending_collect, pending_learn = 0.0, 0.0
            if self.current_learning_iteration % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.logger.stop_logging_writer()

    def _after_update(self, iteration):
        pass

    def _update_world_models(self, batches):
        for index in range(next(iter(batches.values())).shape[0]):
            metrics = self._update_world_model({key: value[index] for key, value in batches.items()})
        return metrics

    def _update_world_model(self, window):
        loss = self.world_model.loss(window)
        self.wm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm(self.world_model.parameters(), 1.0)
        self.wm_optimizer.step()
        return {"world_model": loss.detach()}

    def _update_policy(self, start):
        return self.alg.update(start)

    def save(self, path, infos=None):
        checkpoint = dict(
            format_version=1,
            actor_state_dict=(self.actor_ema if self.actor_ema is not None else self.actor).state_dict(),
            critic_state_dict=self.critic.state_dict(),
            world_model_state_dict=self.world_model.state_dict(),
            actor_optimizer_state_dict=self.alg.actor_optimizer.state_dict(),
            critic_optimizer_state_dict=self.alg.critic_optimizer.state_dict(),
            world_model_optimizer_state_dict=self.wm_optimizer.state_dict(),
            iter=self.current_learning_iteration,
            sim_steps=self.sim_steps,
            collect_fraction=self._collect_fraction,
            cfg=self.cfg,
            task_metadata=getattr(self.task, "metadata", {}),
            infos={**self.checkpoint_infos, **(infos or {})},
        )
        if self.actor_ema is not None:
            checkpoint["actor_training_state_dict"] = self.actor.state_dict()
            checkpoint["actor_ema_updates"] = self._actor_ema_updates
        torch.save(checkpoint, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Expected a ModelBasedRunner checkpoint")
        actor_state = (
            checkpoint.get("actor_training_state_dict", checkpoint["actor_state_dict"])
            if self.training else checkpoint["actor_state_dict"]
        )
        self.actor.load_state_dict(actor_state, strict=strict)
        if self.training:
            if self.actor_ema is not None:
                self.actor_ema.load_state_dict(checkpoint["actor_state_dict"], strict=strict)
                self._actor_ema_updates = checkpoint.get("actor_ema_updates", 1)
            self._check_task_metadata(checkpoint)
            self.critic.load_state_dict(checkpoint["critic_state_dict"], strict=strict)
            self.world_model.load_state_dict(checkpoint["world_model_state_dict"], strict=strict)
            self.alg.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.alg.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.wm_optimizer.load_state_dict(checkpoint["world_model_optimizer_state_dict"])
            if self.cfg.get("capture_updates", False):
                self.wm_optimizer = make_capturable(self.wm_optimizer)
                self.alg.actor_optimizer = make_capturable(self.alg.actor_optimizer)
                self.alg.critic_optimizer = make_capturable(self.alg.critic_optimizer)
                for update in (self._update_world_model, self._update_world_models, self._update_policy):
                    if isinstance(update, GraphedUpdate):
                        update.invalidate()
        self.current_learning_iteration = checkpoint["iter"]
        self.sim_steps = checkpoint.get("sim_steps", 0)
        self.logger.tot_timesteps = self.sim_steps
        self._collect_fraction = checkpoint.get("collect_fraction", 0.0)
        self.checkpoint_infos = checkpoint.get("infos") or {}
        return self.checkpoint_infos

    def _check_task_metadata(self, checkpoint):
        if checkpoint.get("task_metadata", {}) != getattr(self.task, "metadata", {}):
            raise ValueError("World-model task metadata differs (asset, state/context convention or control period)")

    def get_inference_policy(self, device=None):
        actor = self.actor_ema if self.actor_ema is not None else self.actor
        return actor.eval().to(device or self.device)

    @torch.no_grad()
    def _update_actor_ema(self):
        if self._actor_ema_updates == 0:
            # Include the real normalization statistics accumulated during WM warm-up.
            self.actor_ema.load_state_dict(self.actor.state_dict())
            self._actor_ema_updates = 1
            return
        averaged, current = list(self.actor_ema.parameters()), list(self.actor.parameters())
        for avg, live in zip(self.actor_ema.buffers(), self.actor.buffers()):
            if avg.is_floating_point():
                averaged.append(avg)
                current.append(live)
            else:
                avg.copy_(live)  # Normalization sample counts are integers.
        torch._foreach_lerp_(averaged, current, 1 - self.actor_ema_decay)
        self._actor_ema_updates += 1

    def add_git_repo_to_log(self, path):
        self.logger.git_status_repos.append(path)

    def export_policy_to_onnx(self, path, filename="policy.onnx", verbose=False):
        actor = self.actor_ema if self.actor_ema is not None else self.actor
        model = actor.as_onnx(verbose).cpu().eval()
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

    def export_policy_to_jit(self, path, filename="policy.pt"):
        actor = self.actor_ema if self.actor_ema is not None else self.actor
        model = actor.as_onnx(False).cpu().eval()
        os.makedirs(path, exist_ok=True)
        torch.jit.trace(model, model.get_dummy_inputs()).save(os.path.join(path, filename))
