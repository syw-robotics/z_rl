# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch
from z_rl.env import VecEnv
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv


class ZRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for the Z-RL library

    .. caution::
        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        clip_actions: float | None = None,
        obs_group_concat_mode: str = "term_major",
    ):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the Z-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.
            obs_group_concat_mode: Concatenation layout for observation groups that are returned as
                term dictionaries. Supported values are ``"term_major"`` and ``"history_major"``.
                If set to ``"history_major"``, self.concatenate_terms has to be set to False in
                IsaacLab's ManagerBasedRLEnv ObservationCfg.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.
        """

        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise ValueError(f"The environment must be inherited from ManagerBasedRLEnv. Environment type: {type(env)}")

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions
        if obs_group_concat_mode not in ("term_major", "history_major"):
            raise ValueError(
                "obs_group_concat_mode must be either 'term_major' or 'history_major'."
                f" Received: '{obs_group_concat_mode}'."
            )
        self.obs_group_concat_mode = obs_group_concat_mode

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # modify the action space to the clip range
        self._modify_action_space()

        # cache static observation metadata used by downstream RL code
        self._obs_format = self._create_obs_format()
        self._obs_group_layout_mode_map = self._create_obs_group_layout_mode_map()
        self._obs_group_term_order_map = self._create_obs_group_term_order_map()
        self._obs_group_concatenate_dim_map = self._create_obs_group_concatenate_dim_map()
        self._dict_obs_group_names = self._create_dict_obs_group_names()
        self._history_major_group_history_length_map = self._create_history_major_group_history_length_map()
        self._obs_group_time_slice_map = self._create_obs_group_time_slice_map()

        # reset at the start since the Z-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @property
    def obs_format(self) -> dict[str, dict[str, tuple[int, ...]]]:
        """The observation term format for each observation group.

        The returned structure is:

        .. code-block:: python

            {
                "group_name": {
                    "term_name": (history_length, *single_frame_shape),
                },
            }

        The first element of each tuple is always the configured history length.
        The remaining elements correspond to the single-frame term shape.
        """
        return self._obs_format

    @property
    def obs_group_layout_mode_map(self) -> dict[str, str]:
        """The configured layout mode for each observation group."""
        return self._obs_group_layout_mode_map

    @property
    def obs_group_time_slice_map(self) -> dict[str, dict[str, slice | torch.Tensor]]:
        """The cached time-slice selectors for compatible observation groups."""
        return self._obs_group_time_slice_map

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in Z-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        # reset the environment
        obs_dict, extras = self.env.reset()
        return TensorDict(self._process_obs_groups(obs_dict), batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment."""
        obs_dict = self.unwrapped.observation_manager.compute()
        return TensorDict(self._process_obs_groups(obs_dict), batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with Z-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # return the step information
        return TensorDict(self._process_obs_groups(obs_dict), batch_size=[self.num_envs]), rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _create_obs_format(self) -> dict[str, dict[str, tuple[int, ...]]]:
        """Creates the cached observation format metadata."""
        obs_manager = self.unwrapped.observation_manager
        return {
            group_name: {
                term_name: ((term_cfg.history_length,) + self._get_single_frame_obs_term_dim(term_dim, term_cfg))
                for term_name, term_dim, term_cfg in zip(
                    obs_manager.active_terms[group_name],
                    obs_manager.group_obs_term_dim[group_name],
                    obs_manager._group_obs_term_cfgs[group_name],
                )
            }
            for group_name in obs_manager.active_terms
        }

    def _create_obs_group_time_slice_map(self) -> dict[str, dict[str, slice | torch.Tensor]]:
        """Creates cached group-level time slices for concatenated vector observation groups."""
        obs_manager = self.unwrapped.observation_manager
        obs_group_time_slice_map = {}
        for group_name in obs_manager.active_terms:
            if not self._is_obs_group_time_slice_compatible(group_name, obs_manager):
                continue
            layout_mode = self._obs_group_layout_mode_map[group_name]
            obs_group_time_slice_map[group_name] = self._build_obs_group_time_slices(group_name, layout_mode)
        if obs_group_time_slice_map:
            print(
                "[ZRlVecEnvWrapper]: enabled obs time slicing for groups: "
                f"{[(group_name, self._obs_group_layout_mode_map[group_name]) for group_name in obs_group_time_slice_map]}"
            )
        return obs_group_time_slice_map

    def _create_obs_group_layout_mode_map(self) -> dict[str, str]:
        """Creates the wrapper output layout for each observation group."""
        obs_manager = self.unwrapped.observation_manager
        return {
            group_name: self._get_obs_group_layout_mode(group_name, obs_manager)
            for group_name in obs_manager.active_terms
        }

    def _create_obs_group_term_order_map(self) -> dict[str, tuple[str, ...]]:
        """Creates cached term order metadata for each observation group."""
        obs_manager = self.unwrapped.observation_manager
        return {group_name: tuple(obs_manager.active_terms[group_name]) for group_name in obs_manager.active_terms}

    def _create_obs_group_concatenate_dim_map(self) -> dict[str, int]:
        """Creates cached concatenation dimensions for observation groups."""
        obs_manager = self.unwrapped.observation_manager
        return {
            group_name: obs_manager._group_obs_concatenate_dim[group_name]
            for group_name in obs_manager._group_obs_concatenate_dim
        }

    def _create_dict_obs_group_names(self) -> tuple[str, ...]:
        """Creates the cached list of observation groups that require dict-to-tensor conversion."""
        obs_manager = self.unwrapped.observation_manager
        return tuple(
            group_name for group_name in obs_manager.active_terms if not obs_manager.group_obs_concatenate[group_name]
        )

    def _create_history_major_group_history_length_map(self) -> dict[str, int]:
        """Creates cached history lengths for history-major observation groups."""
        return {
            group_name: next(iter({term_format[0] for term_format in term_formats.values()}))
            for group_name, term_formats in self._obs_format.items()
            if self._obs_group_layout_mode_map[group_name] == "history_major"
        }

    def _is_obs_group_time_slice_compatible(self, group_name: str, obs_manager) -> bool:
        """Checks whether an observation group supports cached time slicing."""
        group_dim = obs_manager.group_obs_dim[group_name]
        term_formats = self._obs_format[group_name]
        history_lengths = {term_format[0] for term_format in term_formats.values()}
        # Require all terms to have the same non-zero history length.
        if 0 in history_lengths or len(history_lengths) != 1:
            return False

        if obs_manager.group_obs_concatenate[group_name]:
            # Concatenated groups must already be flat vectors.
            if not isinstance(group_dim, tuple) or len(group_dim) != 1:
                return False
        else:
            # Dict groups must be vector terms so the wrapper can concatenate them.
            if any(len(term_format[1:]) != 1 for term_format in term_formats.values()):
                return False

        for term_format in term_formats.values():
            frame_shape = term_format[1:]
            # Only support vector terms per frame.
            if len(frame_shape) != 1:
                return False
        return True

    def _get_obs_group_layout_mode(self, group_name: str, obs_manager) -> str:
        """Returns the wrapper output layout for an observation group."""
        if (
            not obs_manager.group_obs_concatenate[group_name]
            and self.obs_group_concat_mode == "history_major"
            and self._is_obs_group_time_slice_compatible(group_name, obs_manager)
        ):
            return "history_major"
        return "term_major"

    def _build_obs_group_time_slices(self, group_name: str, layout_mode: str) -> dict[str, slice | torch.Tensor]:
        """Builds cached time-slice selectors for a validated observation group."""
        term_formats = self._obs_format[group_name]
        common_history_length = next(iter({term_format[0] for term_format in term_formats.values()}))
        if layout_mode == "history_major":
            frame_dim = sum(term_format[1] for term_format in term_formats.values())
            group_dim = common_history_length * frame_dim
            return {
                "last": slice(group_dim - frame_dim, group_dim),
                "exclude_last": slice(0, group_dim - frame_dim),
                "exclude_first": slice(frame_dim, group_dim),
            }

        flat_offset = 0
        last_indices = []
        exclude_last_indices = []
        exclude_first_indices = []
        for term_format in term_formats.values():
            frame_dim = term_format[1]
            frame_slices = [
                slice(flat_offset + frame_idx * frame_dim, flat_offset + (frame_idx + 1) * frame_dim)
                for frame_idx in range(common_history_length)
            ]
            last_indices.extend(range(frame_slices[-1].start, frame_slices[-1].stop))
            for frame_slice in frame_slices[:-1]:
                exclude_last_indices.extend(range(frame_slice.start, frame_slice.stop))
            for frame_slice in frame_slices[1:]:
                exclude_first_indices.extend(range(frame_slice.start, frame_slice.stop))
            flat_offset += common_history_length * frame_dim

        return {
            "last": torch.tensor(last_indices, device=self.device, dtype=torch.long),
            "exclude_last": torch.tensor(exclude_last_indices, device=self.device, dtype=torch.long),
            "exclude_first": torch.tensor(exclude_first_indices, device=self.device, dtype=torch.long),
        }

    def _process_obs_groups(self, obs_dict: dict) -> dict:
        """Converts dict-based observation groups into tensors using the configured layout."""
        if not self._dict_obs_group_names:
            return obs_dict
        processed_obs_dict = obs_dict.copy()
        for group_name in self._dict_obs_group_names:
            processed_obs_dict[group_name] = self._concatenate_obs_group(group_name, processed_obs_dict[group_name])
        return processed_obs_dict

    def _concatenate_obs_group(self, group_name: str, group_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenates a dict-based observation group into a tensor."""
        if self._obs_group_layout_mode_map[group_name] == "history_major":
            return self._concatenate_obs_group_history_major(group_name, group_obs)
        return self._concatenate_obs_group_term_major(group_name, group_obs)

    def _concatenate_obs_group_term_major(self, group_name: str, group_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenates group terms in Isaac Lab's default term-major layout."""
        return torch.cat(
            [group_obs[term_name] for term_name in self._obs_group_term_order_map[group_name]],
            dim=self._obs_group_concatenate_dim_map[group_name],
        )

    def _concatenate_obs_group_history_major(self, group_name: str, group_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenates vector history terms into a history-major flat layout."""
        term_order = self._obs_group_term_order_map[group_name]
        common_history_length = self._history_major_group_history_length_map[group_name]
        reshaped_terms = [
            group_obs[term_name].reshape(
                self.num_envs, common_history_length, self._obs_format[group_name][term_name][1]
            )
            for term_name in term_order
        ]
        return torch.cat(reshaped_terms, dim=-1).reshape(self.num_envs, -1)

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other
        #   action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )

    @staticmethod
    def _get_single_frame_obs_term_dim(term_dim: tuple[int, ...], term_cfg) -> tuple[int, ...]:
        """Returns the single-frame observation shape for a term."""
        if term_cfg.history_length == 0:
            return term_dim
        if term_cfg.flatten_history_dim:
            if len(term_dim) != 1:
                raise ValueError(
                    "Flattened history observation terms are expected to have a single non-batch dimension."
                    f" Received term_dim={term_dim}."
                )
            return (term_dim[0] // term_cfg.history_length,)
        return term_dim[1:]
