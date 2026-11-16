from types import SimpleNamespace

import torch
from tensordict import TensorDict
from z_rl.runners.model_based_runner import ModelBasedRunner


class Env:
    num_envs = 4
    num_actions = 2
    device = "cpu"
    cfg = SimpleNamespace()

    @property
    def unwrapped(self):
        return self

    @property
    def model_based_task(self):
        raise AssertionError("Inference must not build the training task/FK")

    def get_observations(self):
        return TensorDict(dict(policy=torch.ones(4, 3)), batch_size=[4])


class ToyTask(torch.nn.Module):
    num_joints, context_dim, num_frames, max_episode_length = 2, 1, 20, 5
    metadata = dict(control_dt=0.02, state="toy")

    def observations(self, state, previous_action, context, frame):
        return TensorDict(dict(policy=state[:, 13:15], critic=state[:, 13:15]), batch_size=state.shape[:1])

    def add_noise(self, obs):
        return obs

    def evaluate(self, state, frame, action, previous_action, context, anchor_state):
        return (
            self.observations(state, action, context, frame),
            -(state[:, 13:15] - 0.2).square().sum(-1),
            torch.zeros_like(frame, dtype=state.dtype),
        )


class TrainingEnv(Env):
    def __init__(self):
        self.state = torch.zeros(4, 17)
        self.state[:, 6] = 1
        self.age = torch.zeros(4, dtype=torch.long)
        self.previous = torch.zeros(4, 2)
        self.model_based_context = torch.zeros(4, 1)
        self.task = ToyTask()

    @property
    def model_based_task(self):
        return self.task

    def get_observations(self):
        return self.task.observations(self.state, self.previous, self.model_based_context, self.age)

    def step(self, action):
        before = self.state.clone()
        nxt = before.clone()
        nxt[:, 13:15] += 0.05 * action
        done = self.age + 1 >= 5
        transition = dict(
            state=before,
            next_state=nxt,
            action=action,
            previous_action=self.previous.clone(),
            context=self.model_based_context,
            frame=self.age.clone(),
            episode_step=self.age.clone(),
            segment_end=done,
            terminated=torch.zeros_like(done),
            truncated=done,
        )
        self.state = torch.where(done[:, None], before * 0, nxt)
        self.state[:, 6] = 1
        self.previous = action
        self.age = torch.where(done, 0, self.age + 1)
        return self.get_observations(), torch.zeros(4), done, dict(model_based_transition=transition)


def training_cfg():
    return dict(
        actor=dict(class_name="z_rl.models:MLPModel", hidden_dims=[16], activation="elu", obs_normalization=True),
        critic=dict(class_name="z_rl.models:MLPModel", hidden_dims=[16], activation="elu", obs_normalization=True),
        obs_groups=dict(actor=["policy"], critic=["critic", "context"]),
        world_model=dict(class_name="z_rl.models.dynamics:LocalResidualDynamics", hidden_dims=[16]),
        world_model_training=dict(
            learning_rate=0.001, unroll_length=2, batch_size=8, updates_per_iteration=1, trainable=True, checkpoint=None
        ),
        algorithm=dict(
            class_name="z_rl.algorithms.shac:SHAC",
            horizon=3,
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            critic_epochs=1,
            critic_minibatches=1,
        ),
        replay_capacity_steps=8,
        imagination_batch_size=8,
        collect_steps_per_iteration=1.5,
        warmup_steps=3,
        model_warmup_iterations=0,
        action_noise_std=0.1,
        lr_decay_iterations=10,
        save_interval=2,
        logger="tensorboard",
        num_steps_per_env=1,
    )


def test_training_resume_preserves_optimizer_cadence_and_frozen_pretrained_norm(tmp_path):
    cfg = training_cfg()
    runner = ModelBasedRunner(TrainingEnv(), cfg, str(tmp_path / "first"))
    runner.learn(2)
    assert runner.logger.tot_timesteps == runner.sim_steps == 24
    checkpoint = tmp_path / "first/model_2.pt"
    resumed = ModelBasedRunner(TrainingEnv(), cfg, str(tmp_path / "resumed"))
    resumed.load(checkpoint)
    assert resumed.current_learning_iteration == 2 and resumed.sim_steps == runner.sim_steps
    assert resumed.logger.tot_timesteps == runner.sim_steps
    assert resumed._collect_fraction == runner._collect_fraction
    assert resumed.alg.actor_optimizer.param_groups[0]["lr"] == runner.alg.actor_optimizer.param_groups[0]["lr"]
    resumed.learn(1)
    assert (tmp_path / "resumed/model_3.pt").is_file()
    cfg["world_model_training"].update(trainable=False, checkpoint=str(checkpoint))
    frozen = ModelBasedRunner(TrainingEnv(), cfg, str(tmp_path / "frozen"))
    before = {k: v.clone() for k, v in frozen.world_model.state_dict().items()}
    frozen.learn(1)
    for key, value in frozen.world_model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_inference_load_needs_only_actor_and_no_replay_or_world_model(tmp_path):
    cfg = dict(
        actor=dict(class_name="z_rl.models:MLPModel", hidden_dims=[8], activation="elu", init_weights=None),
        obs_groups=dict(actor=["policy"], critic=["critic", "context"]),
    )
    runner = ModelBasedRunner(Env(), cfg, device="cpu")
    path = tmp_path / "model.pt"
    torch.save(dict(format_version=1, actor_state_dict=runner.actor.state_dict(), iter=12, infos={}), path)
    policy = runner.get_inference_policy()
    before = policy(Env().get_observations()).detach().clone()
    runner.load(path)
    torch.testing.assert_close(before, policy(Env().get_observations()))
    assert not hasattr(runner, "replay") and not hasattr(runner, "world_model")
    assert runner.current_learning_iteration == 12


def test_onnx_export_matches_actor(tmp_path):
    import numpy as np
    import onnxruntime as ort

    cfg = dict(
        actor=dict(class_name="z_rl.models:MLPModel", hidden_dims=[8], activation="elu"),
        obs_groups=dict(actor=["policy"]),
    )
    runner = ModelBasedRunner(Env(), cfg)
    runner.export_policy_to_onnx(str(tmp_path))
    session = ort.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    x = np.array([[0.1, -0.2, 0.4]], dtype=np.float32)
    expected = runner.actor(TensorDict(dict(policy=torch.from_numpy(x)), batch_size=[1])).detach().numpy()
    np.testing.assert_allclose(session.run(None, {"obs": x})[0], expected, atol=1e-6)


def test_pretrained_model_rejects_incompatible_task_metadata(tmp_path):
    import pytest

    cfg = training_cfg()
    runner = ModelBasedRunner(TrainingEnv(), cfg, str(tmp_path / "first"))
    runner.save(tmp_path / "model.pt")
    cfg["world_model_training"]["checkpoint"] = str(tmp_path / "model.pt")
    env = TrainingEnv()
    env.task.metadata = dict(control_dt=0.04, state="toy")
    with pytest.raises(ValueError, match="task metadata"):
        ModelBasedRunner(env, cfg, str(tmp_path / "other"))


def test_ema_checkpoint_exports_average_and_resumes_live_weights(tmp_path):
    cfg = {**training_cfg(), "actor_ema_decay": 0.5}
    runner = ModelBasedRunner(TrainingEnv(), cfg, str(tmp_path / "train"))
    runner._actor_ema_updates = 1
    with torch.no_grad():
        for avg, live in zip(runner.actor_ema.parameters(), runner.actor.parameters()):
            avg.zero_()
            live.fill_(2.0)
        for avg, live in zip(runner.actor_ema.buffers(), runner.actor.buffers()):
            if avg.is_floating_point():
                avg.zero_()
                live.fill_(4.0)
            else:
                live.fill_(7)
    runner._update_actor_ema()
    for value in runner.actor_ema.parameters():
        torch.testing.assert_close(value, torch.ones_like(value))
    for value in runner.actor_ema.buffers():
        torch.testing.assert_close(value, torch.full_like(value, 2 if value.is_floating_point() else 7))
    checkpoint = tmp_path / "model.pt"
    runner.save(checkpoint)
    resumed = ModelBasedRunner(TrainingEnv(), cfg, str(tmp_path / "resume"))
    resumed.load(checkpoint)
    for live, expected in zip(resumed.actor.parameters(), runner.actor.parameters()):
        torch.testing.assert_close(live, expected)
    for avg, expected in zip(resumed.actor_ema.parameters(), runner.actor_ema.parameters()):
        torch.testing.assert_close(avg, expected)
    assert resumed._actor_ema_updates == runner._actor_ema_updates
    inference = ModelBasedRunner(TrainingEnv(), cfg)
    inference.load(checkpoint)
    obs = TrainingEnv().get_observations()
    torch.testing.assert_close(inference.get_inference_policy()(obs), runner.get_inference_policy()(obs))


def test_enabling_ema_on_an_existing_checkpoint_initializes_from_live_actor(tmp_path):
    cfg = training_cfg()
    runner = ModelBasedRunner(TrainingEnv(), cfg, str(tmp_path / "train"))
    checkpoint = tmp_path / "model.pt"
    runner.save(checkpoint)
    resumed = ModelBasedRunner(TrainingEnv(), {**cfg, "actor_ema_decay": 0.995}, str(tmp_path / "resume"))
    resumed.load(checkpoint)
    for avg, live in zip(resumed.actor_ema.state_dict().values(), resumed.actor.state_dict().values()):
        torch.testing.assert_close(avg, live)


def test_first_ema_update_copies_normalization_after_warmup(tmp_path):
    cfg = {**training_cfg(), "actor_ema_decay": 0.995}
    runner = ModelBasedRunner(TrainingEnv(), cfg, str(tmp_path / "train"))
    runner.collect(4)
    runner._update_actor_ema()
    assert runner._actor_ema_updates == 1
    for avg, live in zip(runner.actor_ema.state_dict().values(), runner.actor.state_dict().values()):
        torch.testing.assert_close(avg, live)


def test_muon_training_and_optimizer_resume(tmp_path):
    from z_rl.utils.opt import MuonAdamWWrapper

    cfg = training_cfg()
    cfg["algorithm"]["use_muon"] = True
    cfg["world_model_training"]["use_muon"] = True
    runner = ModelBasedRunner(TrainingEnv(), cfg, str(tmp_path / "train"))
    for optimizer in (runner.alg.actor_optimizer, runner.alg.critic_optimizer, runner.wm_optimizer):
        assert isinstance(optimizer, MuonAdamWWrapper)
        assert all(group["weight_decay"] == 0.0 for group in optimizer.param_groups)
    runner.learn(2)
    resumed = ModelBasedRunner(TrainingEnv(), cfg, str(tmp_path / "resume"))
    resumed.load(tmp_path / "train/model_2.pt")
    resumed.learn(1)
    assert resumed.current_learning_iteration == 3


def test_capture_updates_requires_cuda_training_device(tmp_path):
    import pytest

    cfg = training_cfg()
    cfg['capture_updates'] = True
    with pytest.raises(ValueError, match='requires a CUDA training device'):
        ModelBasedRunner(TrainingEnv(), cfg, str(tmp_path))
