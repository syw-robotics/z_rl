"""Short-horizon actor-critic through learned, frozen dynamics."""

from contextlib import contextmanager

import torch
from torch import nn
from z_rl.utils.opt import build_optimizer


@contextmanager
def frozen(*modules):
    parameters = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    for parameter, _ in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, enabled in parameters:
            parameter.requires_grad_(enabled)


def lambda_returns(reward, done, values, gamma, lam, truncated=None):
    target = values[-1]
    targets = []
    for t in range(len(reward) - 1, -1, -1):
        continuation = (1 - lam) * values[t + 1] + lam * target
        if truncated is not None:
            continuation = torch.where(truncated[t], values[t + 1], continuation)
        target = reward[t] + gamma * (1 - done[t]) * continuation
        targets.append(target)
    return torch.stack(targets[::-1])


class SHAC:
    def __init__(
        self,
        actor,
        critic,
        world_model,
        task,
        horizon=32,
        gamma=0.99,
        lam=0.95,
        actor_learning_rate=2e-4,
        critic_learning_rate=2e-4,
        critic_epochs=5,
        critic_minibatches=4,
        max_grad_norm=1.0,
        max_grad_value=0.1,
        clip_actions=None,
        compile=False,
        use_muon=False,  # Legacy override for optimizer="muon_adamw".
        optimizer="adam",
        actor_weight_decay=0.0,
    ):
        self.actor, self.critic, self.world_model, self.task = actor, critic, world_model, task
        self.horizon, self.gamma, self.lam = horizon, gamma, lam
        self.critic_epochs, self.critic_minibatches = critic_epochs, critic_minibatches
        self.max_grad_norm, self.max_grad_value, self.clip_actions = max_grad_norm, max_grad_value, clip_actions
        self.actor_optimizer = build_optimizer(
            [actor], learning_rate=actor_learning_rate, optimizer=optimizer, use_muon=use_muon,
            weight_decay=actor_weight_decay, muon_adamw_weight_decay=0.0,
        )
        self.critic_optimizer = build_optimizer(
            [critic], learning_rate=critic_learning_rate, optimizer=optimizer, use_muon=use_muon,
            weight_decay=0.0, muon_adamw_weight_decay=0.0,
        )
        for group in self.actor_optimizer.param_groups:
            if "betas" in group:
                group["betas"] = (0.9, 0.95)
        self.learning_rate = actor_learning_rate
        # Compile the shared tensor step, leaving rollout scheduling and replay outside.
        self._step = torch.compile(self._imagine_step) if compile else self._imagine_step

    def validate_capture(self):
        raise ValueError("SHAC uses dynamic critic minibatches; set capture_policy_updates=False.")

    def capture_config(self):
        return (self.horizon, self.gamma, self.lam, self.critic_epochs, self.critic_minibatches,
                self.max_grad_norm, self.max_grad_value, self.clip_actions)

    def _imagine_step(self, state, obs, frame, previous_action, context):
        # compute action from actor
        action = self.actor(obs)
        if self.clip_actions is not None:
            action = action.clamp(-self.clip_actions, self.clip_actions)

        # compute next state, reward and done using the frozen world model and task congig
        next_state = self.world_model.step(state, action, context)
        next_obs, reward, done = self.task.evaluate(next_state, frame, action, previous_action, context, state)

        # add noise to obs during imagination rollouts, prevent overfitting
        next_obs = self.task.add_noise(next_obs).set("context", context)
        return next_state, next_obs, reward, done, action

    def update(self, start):
        state, context, frame = start["state"], start["context"], start["frame"]
        previous_action = start["previous_action"]
        frame_end = start.get("frame_end", torch.full_like(frame, self.task.num_frames))
        obs = self._initial_observations(start)
        survival = torch.ones_like(frame, dtype=state.dtype)
        objective = torch.zeros_like(survival)
        active = torch.ones_like(frame, dtype=torch.bool)
        critic_obs, rewards, dones, truncations, masks = [], [], [], [], []
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        with frozen(self.world_model, self.critic):
            for t in range(self.horizon):
                critic_obs.append(obs.detach())
                masks.append(active)
                next_state, next_obs, reward, done, action = self._step(state, obs, frame, previous_action, context)
                truncated = (frame + 1 >= frame_end) | (
                    start["episode_step"] + t + 1 >= self.task.max_episode_length
                )
                # bptt objective
                objective = objective + self.gamma**t * survival * reward

                # add critic value for non-terminal states, not for truncated states
                boundary = truncated if t + 1 < self.horizon else torch.ones_like(truncated)
                value = self.critic(next_obs).squeeze(-1)
                objective = objective + self.gamma ** (t + 1) * survival * (1 - done) * boundary * value

                rewards.append(reward.detach())
                dones.append(done.detach())
                truncations.append(truncated)
                survival = survival * (1 - done) * (~truncated)
                active = active & ~truncated
                state = torch.where(active[:, None], next_state, state)

                obs, previous_action = next_obs, action
                frame = torch.minimum(frame + 1, frame_end - 1)

            critic_obs.append(obs.detach())

            loss = -objective.mean()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            if self.max_grad_value is not None:
                nn.utils.clip_grad_value_(self.actor.parameters(), self.max_grad_value)
            self.actor_optimizer.step()

        with torch.no_grad():
            values = torch.stack([self.critic(x).squeeze(-1) for x in critic_obs])
            targets = lambda_returns(
                torch.stack(rewards), torch.stack(dones), values, self.gamma, self.lam, torch.stack(truncations)
            ).flatten()
            valid = torch.stack(masks).flatten().nonzero().squeeze(-1)
        inputs = torch.cat(critic_obs[:-1], 0)
        value_loss = state.new_zeros(())
        for _ in range(self.critic_epochs):
            permutation = valid[torch.randperm(len(valid), device=state.device)]
            for indices in permutation.tensor_split(min(self.critic_minibatches, len(valid))):
                value_loss = (self.critic(inputs[indices]).squeeze(-1) - targets[indices]).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                self.critic_optimizer.step()
        return dict(
            actor=loss.detach(),
            critic=value_loss.detach(),
            actor_grad_norm=grad.detach(),
            imagined_reward=torch.stack(rewards).mean(),
            survival=survival.detach().mean(),
        )

    def _initial_observations(self, start):
        return self.task.add_noise(self.task.observations(
            start["state"], start["previous_action"], start["context"], start["frame"]
        )).set("context", start["context"])
