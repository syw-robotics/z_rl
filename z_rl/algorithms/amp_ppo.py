from z_rl.algorithms.ppo import PPO
from z_rl.modules import AMPModule


class AMPPPO(PPO):
    def __init__(self, *args, **kwargs):
        amp_kwargs = {
            "amp_policy_obs_group": kwargs.pop("amp_policy_obs_group", "amp_policy"),
            "amp_reference_obs_group": kwargs.pop("amp_reference_obs_group", "amp_reference"),
            "amp_reward_coef": kwargs.pop("amp_reward_coef", 1.0),
            "task_reward_lerp": kwargs.pop("amp_task_reward_lerp", 0.0),
            "amp_loss_coef": kwargs.pop("amp_loss_coef", 1.0),
            "amp_grad_penalty_coef": kwargs.pop("amp_grad_penalty_coef", 10.0),
            "hidden_dims": kwargs.pop("amp_discriminator_hidden_dims", [1024, 512]),
            "activation": kwargs.pop("amp_discriminator_activation", "relu"),
            "learning_rate": kwargs.pop("amp_discriminator_learning_rate", 1.0e-3),
            "optimizer": kwargs.pop("amp_discriminator_optimizer", "adam"),
        }

        super().__init__(*args, **kwargs)

        self.amp = AMPModule(
            num_envs=self.storage.num_envs,
            num_steps_per_env=self.storage.num_transitions_per_env,
            device=self.device,
            **amp_kwargs,
        )

    def act(self, obs):
        self.amp.on_act(obs)
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        rewards = self.amp.process_env_step(obs, rewards, dones, extras)
        return super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        losses = super().update()
        amp_losses = self.amp.update(self.num_mini_batches, self.num_learning_epochs)
        losses.update(amp_losses)
        return losses

    def save(self):
        saved_dict = super().save()
        saved_dict.update(self.amp.save())
        return saved_dict

    def load(self, loaded_dict, load_cfg=None, strict=True):
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        self.amp.load(loaded_dict)
        return load_iteration