from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from z_rl.modules import SquashedGaussianDistribution
from z_rl.runners import OnPolicyRunner


class ContinuousEnv:
    num_envs, num_actions, device = 4, 2, "cpu"
    cfg = {}
    single_action_space = SimpleNamespace(low=[-2.0, 0.5], high=[1.0, 4.5])

    def __init__(self):
        self.obs = TensorDict({"policy": torch.zeros(4, 1), "priv": torch.arange(4.0)[:, None]}, [4])

    def get_observations(self):
        return self.obs

    def step(self, actions):
        assert (actions >= torch.tensor(self.single_action_space.low)).all()
        assert (actions <= torch.tensor(self.single_action_space.high)).all()
        self.obs["policy"].add_(1)
        return self.obs, -actions.square().sum(-1), torch.zeros(4), {}


def runner(*, actor_cfg=None, critic_cfg=None, **algorithm_cfg):
    cfg = {
        "num_steps_per_env": 1,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "priv"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [8], "obs_normalization": True, **(actor_cfg or {})},
        "critic": {"class_name": "SACCriticModel", "hidden_dims": [8], "obs_normalization": True, **(critic_cfg or {})},
        "algorithm": {
            "class_name": "SAC",
            "batch_size": 16,
            "learning_starts": 8,
            "replay_buffer_size": 32,
            **algorithm_cfg,
        },
    }
    return OnPolicyRunner(ContinuousEnv(), cfg)


def test_composable_actor_and_critic_train_encoders_and_update_target_heads():
    spec = {"class_name": "GroupMLPLatentSpec", "encoder_cfgs": {"policy": {"output_dim": 4, "hidden_dims": [8]}}}
    train = runner(
        actor_cfg={"class_name": "ComposableModel", "latent_spec": spec},
        critic_cfg={
            "latent_spec": {**spec, "encoder_cfgs": {**spec["encoder_cfgs"], "priv": {"output_dim": 2}}},
            "head_spec": {"class_name": "SimBaHeadSpec", "hidden_dim": 8, "num_blocks": 1},
        },
    )
    actor, critic = train.alg.actor, train.alg.critic
    encoders = [model.latent_adapter.encoders["policy"] for model in (actor, critic)]
    before = [[p.detach().clone() for p in encoder.parameters()] for encoder in encoders]
    target = critic.target_heads[0].output.weight.detach().clone()
    train.learn(2)
    assert train.alg.update_step == 1
    for encoder, old in zip(encoders, before):
        assert any(not torch.equal(p, saved) for p, saved in zip(encoder.parameters(), old))
    torch.testing.assert_close(
        critic.target_heads[0].output.weight, target.lerp(critic.head[0].output.weight, train.alg.tau)
    )
    assert all(p.grad is None for p in critic.target_heads.parameters())


def test_runner_warmup_updates_targets_and_resumes(tmp_path):
    train = runner()
    alg = train.alg
    old_actor = alg.actor.head[0].weight.detach().clone()
    old_target = alg.critic.target_heads[0][0].weight.detach().clone()
    old_alpha = alg.log_alpha.detach().clone()
    train.learn(1)
    assert alg.update_step == 0
    train.learn(1)
    assert alg.update_step == 1
    assert not torch.equal(old_actor, alg.actor.head[0].weight)
    assert not torch.equal(old_alpha, alg.log_alpha)
    torch.testing.assert_close(
        alg.critic.target_heads[0][0].weight, old_target.lerp(alg.critic.head[0][0].weight, alg.tau)
    )
    assert not torch.equal(old_target, alg.critic.target_heads[0][0].weight)
    assert alg.critic.target_heads[0][0].weight.grad is None
    assert not torch.equal(alg.critic.head[0][0].weight, alg.critic.head[1][0].weight)

    file = tmp_path / "sac.pt"
    train.save(str(file), infos={"task": "continuous"})
    resume = runner()
    assert resume.load(str(file)) == {"task": "continuous"}
    assert resume.alg.update_step == 1 and resume.alg.replay.size == 0
    assert resume.current_learning_iteration == train.current_learning_iteration
    for name in ("actor", "critic"):
        for key, tensor in getattr(alg, name).state_dict().items():
            torch.testing.assert_close(tensor, getattr(resume.alg, name).state_dict()[key])
    for name in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
        torch.testing.assert_close(getattr(alg, name).state_dict(), getattr(resume.alg, name).state_dict())
    torch.testing.assert_close(resume.alg.log_alpha, alg.log_alpha)
    resume.learn(1)
    assert resume.alg.update_step == 1
    resume.learn(1)
    assert resume.alg.update_step == 2
    torch.testing.assert_close(alg.actor.head[0].weight, resume.alg.actor.head[0].weight)
    resume.learn(1)
    assert not torch.equal(alg.actor.head[0].weight, resume.alg.actor.head[0].weight)


@pytest.mark.parametrize("terminal_obs", [True, False])
def test_n_step_replay_handles_wrap_timeouts_and_reused_observations(terminal_obs):
    train = runner(n_steps=3, replay_buffer_size=12, batch_size=256, gamma=0.5)
    alg, obs = train.alg, train.env.obs
    for t in range(5):
        alg.act(obs)
        obs["policy"].fill_(t + 1)
        done = torch.tensor([t == 2, t == 3, t == 3, False])
        extras = {
            "time_outs": torch.tensor([False, t == 3, t == 3, False]),
            "terminated": torch.tensor([t == 2, False, t == 3, False]),
        }
        if terminal_obs:
            extras["time_outs_obs"] = obs.clone()
            extras["time_outs_obs"]["policy"].fill_(100 + t)
        alg.process_env_step(obs, torch.full((4,), float(t)), done, extras)

    batch = alg._sample_batch()
    obs, reward, next_obs, discount = (batch[key] for key in ("obs", "rewards", "next_obs", "discount"))
    ids = obs["priv"].squeeze(1).long()
    assert ids.unique().numel() == 4
    torch.testing.assert_close(obs["policy"].flatten(), torch.full_like(reward, 2))
    torch.testing.assert_close(reward, torch.tensor([2, 3.5, 3.5, 4.5])[ids])
    torch.testing.assert_close(discount, torch.tensor([0.0, 0.25 if terminal_obs else 0.0, 0.0, 0.125])[ids])
    torch.testing.assert_close(
        next_obs["policy"].flatten(), torch.tensor([3.0, 103.0 if terminal_obs else 4, 4, 5])[ids]
    )


def test_squashed_log_density_affine_bounds_and_export():
    dist = SquashedGaussianDistribution(2, action_low=[-2, 0.5], action_high=[1, 4.5])
    head_output = torch.tensor([[[0.2, -0.3], [-0.4, -0.8]]], requires_grad=True)
    dist.update(head_output)
    action, logp = dist.rsample_with_log_prob()
    transforms = [
        torch.distributions.TanhTransform(),
        torch.distributions.AffineTransform(torch.tensor([-0.5, 2.5]), torch.tensor([1.5, 2.0])),
    ]
    normal = torch.distributions.Normal(head_output[:, 0], head_output[:, 1].exp())
    expected = torch.distributions.TransformedDistribution(normal, transforms)
    torch.testing.assert_close(logp, expected.log_prob(action).sum(-1))
    torch.testing.assert_close(dist.log_prob(action), logp)
    torch.testing.assert_close(
        dist.as_deterministic_output_module()(head_output), dist.deterministic_output(head_output)
    )
    extreme = torch.tensor([[[100.0, -100.0], [0.0, 0.0]]], requires_grad=True)
    dist.update(extreme)
    _, logp = dist.rsample_with_log_prob()
    logp.sum().backward()
    assert logp.isfinite().all() and extreme.grad.isfinite().all()


def test_fixed_temperature_and_policy_export(tmp_path):
    import onnx
    from onnx.reference import ReferenceEvaluator

    train = runner(auto_alpha=False, updates_per_iteration=3)
    train.learn(2)
    assert train.alg.update_step == 3 and train.alg.alpha_optimizer is None
    assert train.alg.log_alpha.exp().item() == pytest.approx(0.05)
    obs = train.env.get_observations()[:1]
    policy = train.get_inference_policy()
    expected = policy(obs).detach()
    train.export_policy_to_onnx(str(tmp_path))
    model = onnx.load(tmp_path / "policy.onnx")
    onnx.checker.check_model(model)
    (exported,) = ReferenceEvaluator(model).run(None, {"obs": obs["policy"].numpy()})
    torch.testing.assert_close(torch.from_numpy(exported), expected)


def _distributed_update(rank, world_size, store_file):
    from tests.algorithms.test_distributed import init_gloo

    init_gloo(rank, world_size, store_file)
    try:
        torch.manual_seed(11 + rank)
        train = runner()
        alg = train.alg
        alg.world_size = world_size
        alg.broadcast_parameters()
        train.env.obs["policy"].add_(rank)
        train.learn(4)
        for tensor in (*alg.actor.state_dict().values(), *alg.critic.state_dict().values(), alg.log_alpha.detach()):
            peers = [torch.empty_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(peers, tensor)
            torch.testing.assert_close(peers[0], peers[1])
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Requires CPU Gloo")
def test_sac_distributed_updates_stay_in_sync():
    from tests.algorithms.test_distributed import spawn_gloo

    spawn_gloo(_distributed_update)
