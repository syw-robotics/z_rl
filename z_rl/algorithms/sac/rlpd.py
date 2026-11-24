"""Online SAC updates with a fixed offline rollout and live environment replay."""

from __future__ import annotations

import torch

from z_rl.storage.prior_rollout import load_prior_rollout

from .sac import SAC


class RLPD(SAC):
    """Original RLPD sampling and ensemble objectives with the Z-RL SAC actor."""

    critic_defaults = dict(num_qs=10, num_min_qs=2, layer_norm="pre_activation")
    exclude_initial = False

    def __init__(
        self,
        *args,
        prior_data=None,
        prior_data_ratio=0.5,
        updates_per_iteration=20,
        policy_frequency=None,
        gamma=0.99,
        target_entropy_scale=0.5,
        alpha=1.0,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        **kwargs,
    ):
        super().__init__(
            *args,
            updates_per_iteration=updates_per_iteration,
            gamma=gamma,
            policy_frequency=updates_per_iteration if policy_frequency is None else policy_frequency,
            target_entropy_scale=target_entropy_scale,
            alpha=alpha,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            alpha_learning_rate=alpha_learning_rate,
            **kwargs,
        )
        self.online_count, self.prior_count = self._batch_sizes(self.batch_size, prior_data_ratio)
        if prior_data_ratio > 0 and self.prior_count == 0:
            raise ValueError("The requested prior_data_ratio gives zero prior samples per batch.")
        self.prior_data = prior_data
        self.prior_replay = None
        self.prior_action_key = "action"
        self.prior_reward_keys = None
        self.bootstrap_timeouts = True
        self.bc_gamma = None

    @staticmethod
    def _batch_sizes(batch_size, ratio):
        if not 0 <= ratio <= 1:
            raise ValueError("RLPD prior_data_ratio must be in [0, 1].")
        prior = int(batch_size * ratio)
        return batch_size - prior, prior

    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        critic = {**cls.critic_defaults, **cfg["critic"]}
        alg = super().construct_algorithm(obs, env, {**cfg, "critic": critic}, device)
        alg.prior_action_key = getattr(env, "action_key", "action")
        alg.prior_reward_keys = getattr(env, "reward_keys", None)
        alg.bootstrap_timeouts = not getattr(env.cfg, "is_finite_horizon", False)
        return alg

    def _load_prior(self):
        if self.prior_count and self.prior_replay is None:
            if not self.prior_data:
                raise ValueError("Set algorithm.prior_data to a rollout archive before training with prior data.")
            self.prior_replay = load_prior_rollout(
                self.prior_data,
                {**self.actor.obs_group_shapes, **self.critic.obs_group_shapes},
                self.critic.num_actions,
                action_key=self.prior_action_key,
                reward_keys=self.prior_reward_keys,
                bootstrap_timeouts=self.bootstrap_timeouts,
                bc_gamma=self.bc_gamma,
                n_steps=self.n_steps,
                include_is_init=self.exclude_initial,
            )

    def train_mode(self):
        super().train_mode()
        self._load_prior()

    def _store_transition(self, transition, extras):
        reward = transition["rewards"]
        discount = extras.get("discount")
        transition["env_discount"] = (
            torch.ones_like(reward) if discount is None else discount.to(reward).reshape_as(reward)
        )
        self.replay.add(transition)

    def _ready_to_update(self):
        return not self.online_count or super()._ready_to_update()

    def _sample_batch(self):
        self._load_prior()
        if not self.prior_count:
            return self._sample_replay(self.replay, self.online_count)
        prior = self._sample_replay(self.prior_replay, self.prior_count)
        if not self.online_count:
            return prior
        online = self._sample_replay(self.replay, self.online_count)
        if self.bc_gamma is not None:
            # Only prior rows have demonstration returns and lengths.
            for key in ("return_to_go", "steps_to_go"):
                online[key] = prior[key].new_zeros(self.online_count)
        return torch.cat((online, prior))

    def _policy_schedule(self):
        due = (self.update_step + 1) % self.policy_frequency == 0
        return due, due and self.alpha_optimizer is not None

    def _actor_loss(self, batch, action, logp, metrics):
        q = self.critic(batch["obs"], action).mean(-1)
        return (self.log_alpha.detach().exp() * logp - q).mean()


class BTXRLPD(RLPD):
    """BTX additive prior sampling and return-gated behavior cloning on scalar SAC critics."""

    critic_defaults = dict(num_qs=2, num_min_qs=2)
    exclude_initial = True

    def __init__(
        self,
        *args,
        prior_data_ratio=0.4,
        updates_per_iteration=4,
        gamma=0.99,
        n_steps=3,
        alpha=0.004,
        bc_loss="mse",
        bc_coef=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            prior_data_ratio=prior_data_ratio,
            updates_per_iteration=updates_per_iteration,
            gamma=gamma,
            n_steps=n_steps,
            alpha=alpha,
            **kwargs,
        )
        if bc_loss not in ("mse", "nll"):
            raise ValueError("bc_loss must be 'mse' or 'nll'.")
        self.bc_loss = bc_loss
        self.bc_coef = (1.0 if bc_loss == "mse" else 0.05) if bc_coef is None else bc_coef
        if self.bc_coef < 0 or (self.bc_coef and self.prior_count and gamma == 1):
            raise ValueError("Gated BC requires a nonnegative bc_coef and gamma < 1 when enabled.")
        self.bc_gamma = gamma if self.bc_coef and self.prior_count else None
        self._is_init = torch.ones(self.replay.num_envs, dtype=torch.bool, device=self.device)

    @staticmethod
    def _batch_sizes(batch_size, ratio):
        if ratio < 0:
            raise ValueError("BTX prior_data_ratio must be nonnegative.")
        return batch_size, int(batch_size * ratio)

    def _store_transition(self, transition, extras):
        transition["is_init"] = self._is_init
        super()._store_transition(transition, extras)
        self._is_init.copy_(transition["segment_end"])

    def _mean_loss(self, per_row, batch):
        valid = ~batch["is_init"]
        return (per_row * valid).sum() / valid.sum().clamp_min(1)

    def _actor_loss(self, batch, action, logp, metrics):
        # BTX averages four reparameterized actions per state.
        dist = self.actor.distribution
        samples = [(action, logp)] + [dist.rsample_with_log_prob() for _ in range(3)]
        actions, logps = (torch.stack(values) for values in zip(*samples))
        q = self.critic(batch["obs"].expand(4, *batch.batch_size), actions).mean(dim=(0, -1))
        loss = self.log_alpha.detach().exp() * logps.mean(0) - q
        if self.bc_gamma is None:
            return self._mean_loss(loss, batch)

        prior = slice(self.online_count, None)
        demo = batch["actions"][prior]
        # Both Q and the demonstration return use raw Z-RL reward units.
        gap = (1 - self.bc_gamma) * (batch["return_to_go"][prior] - q[prior].detach())
        gate = gap.clamp(0, 1) * (batch["steps_to_go"][prior] > 1 / (1 - self.bc_gamma))
        if self.bc_loss == "mse":
            error = (actions[:, prior] - demo).square().mean(dim=(0, -1))
        else:
            error = -dist.log_prob(batch["actions"])[prior]
        valid = ~batch["is_init"]
        bc = (gate * error * valid[prior]).sum() / valid.sum().clamp_min(1)
        metrics.setdefault("bc_loss", []).append(bc.detach())
        metrics.setdefault("bc_gate_fraction", []).append(((gate > 0) & valid[prior]).float().mean())
        return self._mean_loss(loss, batch) + self.bc_coef * bc
