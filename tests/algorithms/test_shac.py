import torch
from tensordict import TensorDict
from torch import nn
from z_rl.algorithms.shac import SHAC, lambda_returns


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Parameter(torch.tensor([0.1]))

    def forward(self, obs):
        return self.action.expand(obs.batch_size[0], 1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor([1.0]))

    def forward(self, obs):
        return self.slope * obs["policy"]


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))

    def step(self, state, action, context):
        return state + action * self.gain


class Task:
    num_frames = 100
    max_episode_length = 100

    def observations(self, state, previous_action, context, frame):
        return TensorDict(dict(policy=state), batch_size=state.shape[:-1])

    def add_noise(self, obs):
        return obs

    def evaluate(self, state, frame, action, previous_action, context, anchor_state):
        obs = self.observations(state, action, context, frame)
        return obs, torch.zeros_like(frame, dtype=torch.float32), torch.zeros_like(frame, dtype=torch.float32)


def test_actor_gradient_flows_through_frozen_model_and_critic_bootstrap():
    actor, wm, critic = Actor(), Dynamics(), Critic()
    alg = SHAC(actor, critic, wm, Task(), horizon=3, actor_learning_rate=0.01, critic_epochs=0)
    start = dict(
        state=torch.zeros(8, 1),
        previous_action=torch.zeros(8, 1),
        context=torch.zeros(8, 1),
        frame=torch.zeros(8, dtype=torch.long),
        episode_step=torch.zeros(8, dtype=torch.long),
    )
    old = actor.action.clone()
    alg.update(start)
    assert actor.action > old  # reward is zero: only bootstrap can produce this gradient.
    assert wm.gain.grad is None and critic.slope.grad is None
    assert wm.gain.item() == 0.5 and critic.slope.item() == 1


def test_lambda_returns_bootstrap_time_limit_but_not_failure():
    reward = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    done = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    values = torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
    target = lambda_returns(reward, done, values, gamma=0.9, lam=1.0)
    torch.testing.assert_close(target, torch.tensor([[2.8, 10.9], [2.0, 11.0]]))


def test_mixed_clip_ends_bootstrap_once_and_never_advance_into_other_clips():
    class RecordingTask(Task):
        def __init__(self):
            self.frames = []

        def evaluate(self, state, frame, action, previous_action, context, anchor_state):
            self.frames.append(frame.clone())
            return super().evaluate(state, frame, action, previous_action, context, anchor_state)

    task = RecordingTask()
    alg = SHAC(Actor(), Critic(), Dynamics(), task, horizon=4, gamma=1.0, critic_epochs=0)
    result = alg.update(dict(
        state=torch.zeros(2, 1), previous_action=torch.zeros(2, 1), context=torch.zeros(2, 1),
        frame=torch.tensor([0, 5]), frame_end=torch.tensor([1, 8]), episode_step=torch.zeros(2, dtype=torch.long),
    ))
    # First seed has one remaining transition, second has three; bootstrap exactly once.
    torch.testing.assert_close(result["actor"], torch.tensor(-0.1))
    assert torch.stack(task.frames).tolist() == [[0, 5], [0, 6], [0, 7], [0, 7]]
