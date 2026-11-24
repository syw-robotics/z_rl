import pytest
import torch
from tensordict import NonTensorData, TensorDict
from torch import nn

from tests.algorithms.test_sac import runner
from z_rl.storage.prior_rollout import load_prior_rollout


def write_rollout(path, *, compact=False):
    t, n = 8, 3
    obs = torch.arange(t * n, dtype=torch.float32).reshape(t, n, 1) + 80
    done = torch.zeros(t, n, 1, dtype=torch.bool)
    done[3] = True
    terminated = done.clone()
    terminated[:, 1] = False
    truncated = done & ~terminated
    truncated[:, 2] = done[:, 2]  # overlapping flags still terminate
    initial = torch.zeros_like(done)
    initial[0] = initial[4] = True
    td = TensorDict(
        {
            "policy": obs,
            "priv": -obs,
            "is_init": initial,
            "action": torch.tensor([0.25, 2.0]).expand(t, n, 2),
            "next": {
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
                "discount": torch.full_like(obs, 0.5),
                "reward": {"task": torch.full_like(obs, 3.0), "aux": torch.full_like(obs, 2.0)},
            },
        },
        [t, n],
    )
    if not compact:
        td["next", "policy"] = obs + 30
        td["next", "priv"] = -obs - 30
    td["env_meta"] = NonTensorData({"step_dt": 0.02})
    torch.save({"format_version": 1, "stacked": td, "writer_max_size": 50}, path)


@pytest.mark.parametrize("compact", [False, True])
def test_prior_rollout_matches_boundaries_discounts_and_selected_rewards(tmp_path, compact):
    path = tmp_path / "rollout.pt"
    write_rollout(path, compact=compact)
    buf = load_prior_rollout(tmp_path, {"policy": (1,), "priv": (1,)}, 2, reward_keys=["task"], bc_gamma=0.5, n_steps=2)
    assert buf.num_envs == 3 and buf.size == 8 - compact
    data = buf.data
    torch.testing.assert_close(data["rewards"], torch.full((buf.size, 3), 3.0))
    torch.testing.assert_close(data["bootstrap"][3], torch.tensor([False, not compact, False]))
    assert data["steps_to_go"][:4, 0].tolist() == [4, 3, 2, 1]
    assert data["return_to_go"][2, 0].item() == pytest.approx(3 + 0.5 * 0.5 * 3)
    assert data["next_obs"]["policy"][1, 1].item() == 84 + (3 if compact else 30)


@pytest.mark.parametrize(("name", "online", "prior", "heads"), [("RLPD", 8, 8, 10), ("BTXRLPD", 16, 6, 2)])
def test_mixed_update_checkpoint_and_inference_without_dataset(tmp_path, name, online, prior, heads):
    path = tmp_path / "rollout.pt"
    write_rollout(path)
    cfg = dict(class_name=name, prior_data=str(path), updates_per_iteration=2, gamma=0.5, n_steps=2)
    train = runner(**cfg)
    alg = train.alg
    assert alg.prior_replay is None
    assert len(alg.critic.head) == heads
    if name == "RLPD":
        assert any(isinstance(layer, nn.LayerNorm) for layer in alg.critic.head[0])
        assert alg.critic.num_min_qs == 2
    initial_actor = alg.actor.head[0].weight.detach().clone()
    train.learn(4)
    assert alg.update_step == 6
    assert not torch.equal(alg.actor.head[0].weight, initial_actor)
    samples = alg._sample_batch()
    assert len(samples) == online + prior
    assert (samples["obs"]["policy"][:online] < 80).all()
    assert (samples["obs"]["policy"][online:] >= 80).all()
    assert samples["rewards"][:online].max() <= 0
    assert set(samples["rewards"][online:].tolist()).issubset({5.0, 6.25})
    assert alg.prior_replay.size == 8
    ckpt = tmp_path / "model.pt"
    train.save(str(ckpt))
    resumed = runner(**cfg)
    resumed.load(str(ckpt))
    assert resumed.alg.update_step == 6 and resumed.alg.replay.size == 0
    resumed.learn(2)
    assert resumed.alg.update_step == 8
    path.unlink()
    play = runner(**cfg)
    play.load(str(ckpt))
    obs = play.env.get_observations()
    torch.testing.assert_close(play.get_inference_policy()(obs), train.get_inference_policy()(obs))
    with pytest.raises(FileNotFoundError):
        play.learn(1)


@pytest.mark.parametrize("bc_loss", ["mse", "nll"])
def test_btx_bc_gate_uses_prior_only_and_has_policy_gradients(tmp_path, bc_loss):
    path = tmp_path / "prior.pt"
    write_rollout(path)
    train = runner(class_name="BTXRLPD", prior_data=str(path), bc_loss=bc_loss, gamma=0.5, n_steps=1)
    alg = train.alg
    train.learn(2)
    batch = alg._sample_batch()
    for p in alg.critic.head.parameters():
        p.detach().zero_()
    action, logp = alg._sample_policy(batch["obs"])
    batch["is_init"].zero_()
    batch["return_to_go"].fill_(10)
    batch["steps_to_go"].fill_(3)
    # Exclude one short horizon, one worse demonstration, and one initial row.
    batch["steps_to_go"][-1] = 2
    batch["return_to_go"][-2] = -10
    batch["is_init"][-3] = True
    metrics = {}
    torch.manual_seed(8)
    loss = alg._actor_loss(batch, action, logp, metrics)
    assert metrics["bc_gate_fraction"][0].item() == pytest.approx(0.5)
    penalty = metrics["bc_loss"][0]
    assert penalty.abs() > 0
    batch["return_to_go"].fill_(-10)
    torch.manual_seed(8)
    baseline = alg._actor_loss(batch, action, logp, {})
    torch.testing.assert_close(loss - baseline, alg.bc_coef * penalty)
    (loss - baseline).backward()
    assert alg.actor.head[-2].bias.grad.abs().sum() > 0


def test_original_ensemble_random_pair_and_mean_actor():
    alg = runner(class_name="RLPD", prior_data_ratio=0).alg
    obs = TensorDict({"policy": torch.zeros(2, 1), "priv": torch.zeros(2, 1)}, [2])
    action = torch.zeros(2, 2)
    for i, (q, target) in enumerate(zip(alg.critic.head, alg.critic.target_heads)):
        for head in (q, target):
            head[-1].weight.detach().zero_()
            head[-1].bias.detach().fill_(i)
    assert alg._actor_loss({"obs": obs}, action, torch.zeros(2), {}).item() == pytest.approx(-4.5)
    pairs = [alg.critic(obs, action, target=True)[0].tolist() for _ in range(16)]
    assert all(len(set(pair)) == 2 for pair in pairs)
    assert len(set(tuple(sorted(pair)) for pair in pairs)) > 1


def test_prior_only_needs_no_online_warmup(tmp_path):
    path = tmp_path / "prior.pt"
    write_rollout(path)
    alg = runner(class_name="RLPD", prior_data=path, prior_data_ratio=1, updates_per_iteration=1).alg
    assert alg.replay.size == 0 and alg.update()["critic"] > 0


def _distributed_rlpd(rank, world_size, store_file, path):
    from tests.algorithms.test_distributed import init_gloo

    init_gloo(rank, world_size, store_file)
    try:
        for name in ("RLPD", "BTXRLPD"):
            torch.manual_seed(31 + rank)
            train = runner(class_name=name, prior_data=path, updates_per_iteration=2, n_steps=1, gamma=0.5)
            train.alg.world_size = world_size
            train.alg.broadcast_parameters()
            train.env.obs["policy"].add_(rank)
            train.learn(3)
            for tensor in (*train.alg.actor.state_dict().values(), *train.alg.critic.state_dict().values(), train.alg.log_alpha):
                peers = [torch.empty_like(tensor) for _ in range(world_size)]
                torch.distributed.all_gather(peers, tensor.detach())
                torch.testing.assert_close(peers[0], peers[1])
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Requires CPU Gloo")
def test_distributed_mixed_updates_stay_in_sync(tmp_path):
    from tests.algorithms.test_distributed import spawn_gloo

    path = tmp_path / "prior.pt"
    write_rollout(path)
    spawn_gloo(_distributed_rlpd, str(path))
