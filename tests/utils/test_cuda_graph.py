from copy import deepcopy

import pytest
import torch
from z_rl.utils.cuda_graph import GraphedUpdate, graph_state, make_capturable
from z_rl.utils.opt import build_optimizer
from z_rl.runners.model_based_runner import ModelBasedRunner
from z_rl.storage.sequence_replay import SequenceReplayBuffer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_recapture_after_optimizer_load_and_buffer_replacement():
    torch.manual_seed(31)
    eager = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8),
                                torch.nn.Tanh(), torch.nn.Linear(8, 2)).cuda()
    eager.register_buffer('offset', torch.zeros(2, device='cuda'))
    captured = deepcopy(eager)
    models = [eager, captured]
    opts = [make_capturable(build_optimizer([m], learning_rate=.001, use_muon=True,
                                            muon_adamw_weight_decay=0.)) for m in models]

    def update(model, opt, data):
        opt.zero_grad(set_to_none=True)
        loss = (model(data['x']) + model.offset - data['y']).square().mean()
        loss.backward()
        opt.step()
        return {'loss': loss.detach()}

    graph = GraphedUpdate(lambda data: update(captured, opts[1], data),
                          lambda: graph_state([captured], [opts[1]]))
    for step in range(14):
        if step == 5:
            for opt in opts:
                opt.load_state_dict(deepcopy(opt.state_dict()))
        if step == 9:
            for model in models:
                model.offset = torch.full((2,), .5, device='cuda')
        data = dict(x=torch.randn(16, 4, device='cuda'), y=torch.randn(16, 2, device='cuda'))
        expected = update(eager, opts[0], data)
        actual = graph(data)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for a, b in zip(eager.parameters(), captured.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert graph.graph is not None


def test_grouped_updates_match_sequential_optimizer_steps():
    from types import SimpleNamespace
    torch.manual_seed(42)
    first = torch.nn.Linear(4, 2).cuda()
    second = deepcopy(first)
    opts = [make_capturable(torch.optim.Adam(m.parameters(), lr=.001)) for m in (first, second)]

    def update(model, opt, data):
        opt.zero_grad(set_to_none=True)
        loss = (model(data['x']) - data['y']).square().mean()
        loss.backward()
        opt.step()
        return {'world_model': loss.detach()}

    runner = SimpleNamespace(_update_world_model=lambda data: update(second, opts[1], data))
    grouped = GraphedUpdate(lambda data: ModelBasedRunner._update_world_models(runner, data),
                            lambda: graph_state([second], [opts[1]]))
    for _ in range(7):
        data = dict(x=torch.randn(5, 16, 4, device='cuda'), y=torch.randn(5, 16, 2, device='cuda'))
        for i in range(5):
            expected = update(first, opts[0], {k: v[i] for k, v in data.items()})
        torch.testing.assert_close(grouped(data), expected)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b)
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(opts[0].state[a], opts[1].state[b])


def test_selective_replay_matches_full_sampling_and_boundary_mask():
    replay = SequenceReplayBuffer(6, 2, 'cuda')
    for i in range(9):
        replay.add(dict(state=torch.full((2, 3), float(i), device='cuda'),
                        unused=torch.ones(2, 20, device='cuda'),
                        segment_end=torch.full((2,), i % 3 == 0, device='cuda')))
    torch.manual_seed(9)
    full = replay.sample_windows(30, 4)
    torch.manual_seed(9)
    selected = replay.sample_windows(30, 4, fields=('state',))
    assert set(selected) == {'state', 'mask'}
    torch.testing.assert_close(selected['state'], full['state'])
    torch.testing.assert_close(selected['mask'], full['mask'])
