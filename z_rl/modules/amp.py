import torch
import torch.nn.functional as F

from z_rl.modules.discriminator import AMPDiscriminator
from z_rl.storage import AmpStorage
from z_rl.utils import resolve_optimizer


class AMPModule:
    def __init__(
        self,
        num_envs,
        num_steps_per_env,
        device,
        amp_policy_obs_group="amp_policy",
        amp_reference_obs_group="amp_reference",
        amp_reward_coef=1.0,
        task_reward_lerp=0.0,
        amp_loss_coef=1.0,
        amp_grad_penalty_coef=10.0,
        hidden_dims=[1024, 512],
        activation="relu",
        learning_rate=1.0e-3,
        optimizer="adam",
    ):
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.device = device

        self.amp_policy_obs_group = amp_policy_obs_group
        self.amp_reference_obs_group = amp_reference_obs_group

        self.amp_loss_coef = amp_loss_coef
        self.amp_grad_penalty_coef = amp_grad_penalty_coef
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        self.storage = None
        self.discriminator = None
        self.optimizer = None

        self.policy_state = None
        self.reference_state = None

    def on_act(self, obs):
        amp_policy_obs = obs[self.amp_policy_obs_group]
        amp_reference_obs = obs[self.amp_reference_obs_group]

        with torch.inference_mode(False):
            self._ensure_initialized(amp_policy_obs)

        self.policy_state = amp_policy_obs.detach().clone()
        self.reference_state = amp_reference_obs.detach().clone()

    def process_env_step(self, obs, rewards, dones, extras):
        policy_next = obs[self.amp_policy_obs_group].detach()
        reference_next = obs[self.amp_reference_obs_group].detach()

        rewards, amp_logit = self.discriminator.predict_amp_reward(
            self.policy_state,
            policy_next,
            rewards,
        )

        self.storage.add_transition(
            self.policy_state,
            policy_next,
            self.reference_state,
            reference_next,
        )

        extras.setdefault("amp", {})
        extras["amp"]["reward"] = rewards.detach()
        extras["amp"]["policy_logit"] = amp_logit.squeeze(-1).detach()

        self.policy_state = None
        self.reference_state = None
        return rewards

    def update(self, num_mini_batches, num_learning_epochs):
        mean_losses = {}

        for batch in self.storage.mini_batch_generator(num_mini_batches, num_learning_epochs):
            policy_d = self.discriminator(batch.policy_states, batch.policy_next_states)
            expert_d = self.discriminator(batch.reference_states, batch.reference_next_states)
            policy_loss = F.mse_loss(policy_d, -torch.ones_like(policy_d))
            expert_loss = F.mse_loss(expert_d, torch.ones_like(expert_d))
            amp_loss = 0.5 * (policy_loss + expert_loss)

            grad_pen = self.discriminator.compute_grad_pen(
                batch.reference_states,
                batch.reference_next_states,
                lambda_=self.amp_grad_penalty_coef,
            )

            loss = self.amp_loss_coef * (amp_loss + grad_pen)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_losses["amp_loss"] = mean_losses.get("amp_loss", 0.0) + amp_loss.detach()
            mean_losses["amp_grad_penalty"] = mean_losses.get("amp_grad_penalty", 0.0) + grad_pen.detach()

        n = num_mini_batches * num_learning_epochs
        for k in mean_losses:
            mean_losses[k] = mean_losses[k] / n

        self.storage.clear()
        return mean_losses

    def _ensure_initialized(self, amp_obs):
        if self.storage is None:
            self.storage = AmpStorage(
                self.num_envs,
                self.num_steps_per_env,
                amp_obs.shape[1:],
                device=self.device,
            )

        if self.discriminator is None:
            obs_dim = amp_obs.flatten(1).shape[1]
            self.discriminator = AMPDiscriminator(
                input_dim=obs_dim * 2,
                amp_reward_coef=self.amp_reward_coef,
                hidden_dims=self.hidden_dims,
                activation=self.activation,
                task_reward_lerp=self.task_reward_lerp,
            ).to(self.device)

            opt_class = resolve_optimizer(self.optimizer_name)
            self.optimizer = opt_class(self.discriminator.parameters(), lr=self.learning_rate)

    def save(self):
        saved_dict = {}

        if self.discriminator is not None:
            saved_dict["amp_discriminator_state_dict"] = self.discriminator.state_dict()

        if self.optimizer is not None:
            saved_dict["amp_optimizer_state_dict"] = self.optimizer.state_dict()

        return saved_dict

    def load(self, loaded_dict):
        if self.discriminator is not None and "amp_discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"])

        if self.optimizer is not None and "amp_optimizer_state_dict" in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])

