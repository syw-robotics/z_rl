# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from tensordict import TensorDict

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationManager, ObservationTermCfg

from z_rl.env import VecEnv
from z_rl.utils import ObsSelector


def _history_slices(start: int, length: int, width: int, newest_first: bool) -> dict[str, slice]:
    """Slice a contiguous block of frames, retaining their stored order."""
    stop = start + length * width
    without_head = slice(start + width, stop)
    without_tail = slice(start, stop - width)
    return {
        "last": slice(start, start + width) if newest_first else slice(stop - width, stop),
        "exclude_last": without_head if newest_first else without_tail,
        "exclude_first": without_tail if newest_first else without_head,
    }


def _newest_first(term_cfg: ObservationTermCfg) -> bool:
    return term_cfg.history_length > 0 and getattr(term_cfg, "history_newest_first", False)


class ZRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for the Z-RL library

    .. caution::
        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.
    """

    def __init__(self, env: ManagerBasedRLEnv, clip_actions: float | None = None):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the Z-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.
        """

        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise ValueError(f"The environment must be inherited from ManagerBasedRLEnv. Environment type: {type(env)}")

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

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
        obs_manager = self.unwrapped.observation_manager
        self._obs_format = self._create_obs_format(obs_manager)
        self._dict_obs_groups = {
            group_name: (tuple(term_names), obs_manager._group_obs_concatenate_dim[group_name])
            for group_name, term_names in obs_manager.active_terms.items()
            if not obs_manager.group_obs_concatenate[group_name]
        }
        self._obs_group_time_slice_map = self._create_obs_group_time_slice_map(obs_manager)
        self._symmetry_compiler = None

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
    def obs_group_time_slice_map(self) -> dict[str, dict[str, ObsSelector]]:
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
        return self._pack_observations(obs_dict), extras

    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment."""
        obs_dict = self.unwrapped.observation_manager.compute()
        return self._pack_observations(obs_dict)

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
            terminal_obs = extras.get("time_outs_obs", extras.get("final_obs"))
            if terminal_obs is not None:
                # IsaacLab reuses extras across steps. Keep this translated snapshot local to this step.
                extras = {
                    **extras,
                    "terminated": terminated,
                    "time_outs_obs": self._pack_observations(terminal_obs),
                }
        # return the step information
        return self._pack_observations(obs_dict), rew, dones, extras

    def make_command_metric_compute_function(self):
        """Return a function that collects metrics from the active command terms."""
        command_manager = self.unwrapped.command_manager
        term_functions = {}
        for name in command_manager.active_terms:
            make_compute = getattr(command_manager.get_term(name), "compute_command_metrics", None)
            if make_compute is not None:
                compute = make_compute()
                if compute is not None:
                    term_functions[name] = compute
        if not term_functions:
            return None

        def compute_metrics():
            return {
                f"Metrics/{name}/{key}": value
                for name, compute in term_functions.items()
                for key, value in compute().items()
            }

        return compute_metrics

    def compile_symmetry(self) -> None:
        """Compile declared observation/action symmetry once for training."""
        if self._symmetry_compiler is None:
            from .symmetry import IsaacLabSymmetryCompiler

            self._symmetry_compiler = IsaacLabSymmetryCompiler(self)

    def augment_symmetry(
        self,
        obs: TensorDict | None = None,
        actions: torch.Tensor | None = None,
    ) -> tuple[TensorDict | None, torch.Tensor | None]:
        """Append mirrored observations/actions using the cached compiler."""
        self.compile_symmetry()
        return self._symmetry_compiler.augment(obs=obs, actions=actions)

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _create_obs_format(self, obs_manager: ObservationManager) -> dict[str, dict[str, tuple[int, ...]]]:
        """Record each term's history length and single-frame shape."""
        obs_format = {}
        for group, term_names in obs_manager.active_terms.items():
            term_formats = {}
            term_dims = obs_manager.group_obs_term_dim[group]
            term_cfgs = obs_manager._group_obs_term_cfgs[group]
            for name, shape, cfg in zip(term_names, term_dims, term_cfgs):
                history = cfg.history_length
                if history and cfg.flatten_history_dim:
                    if len(shape) != 1:
                        raise ValueError(
                            f"Flattened observation term '{group}/{name}' must have one non-batch dimension, "
                            f"got {shape}."
                        )
                    shape = (shape[0] // history,)
                elif history:
                    shape = shape[1:]
                term_formats[name] = (history, *shape)
            obs_format[group] = term_formats
        return obs_format

    def _create_obs_group_time_slice_map(self, obs_manager: ObservationManager) -> dict[str, dict[str, ObsSelector]]:
        """Precompute temporal selectors for groups of flat vector terms."""
        group_time_major = getattr(obs_manager, "_group_obs_time_major", {})
        time_slices = {}
        for group, term_formats in self._obs_format.items():
            if any(len(shape) != 1 for shape in obs_manager.group_obs_term_dim[group]):
                continue
            formats = tuple(term_formats.values())
            if any(len(fmt) != 2 for fmt in formats):
                continue
            time_slices[group] = self._build_obs_group_time_slices(
                formats, obs_manager._group_obs_term_cfgs[group], time_major=group_time_major.get(group, False)
            )
        if time_slices:
            print(f"[ZRlVecEnvWrapper]: enabled obs time slicing for groups: {list(time_slices)}")
        return time_slices

    def _build_obs_group_time_slices(
        self, formats: tuple[tuple[int, ...], ...], term_cfgs: list[ObservationTermCfg], *, time_major: bool
    ) -> dict[str, ObsSelector]:
        """Build selectors from the group's resolved term formats and history order."""
        if time_major:
            length, _ = formats[0]
            width = sum(width for _, width in formats)
            slices = _history_slices(0, length, width, _newest_first(term_cfgs[0]))
            return {name: ObsSelector(window) for name, window in slices.items()}

        indices = {"last": [], "exclude_last": [], "exclude_first": []}
        offset = 0
        for (history, width), cfg in zip(formats, term_cfgs):
            length = max(history, 1)
            for name, window in _history_slices(offset, length, width, _newest_first(cfg)).items():
                indices[name].extend(range(window.start, window.stop))
            offset += length * width
        return {
            name: ObsSelector(torch.tensor(values, device=self.device, dtype=torch.long))
            for name, values in indices.items()
        }

    def _pack_observations(self, obs_dict: dict) -> TensorDict:
        """Concatenate dictionary groups and wrap observations on the environment device."""
        if self._dict_obs_groups:
            obs_dict = obs_dict.copy()
            for group_name, (term_names, dim) in self._dict_obs_groups.items():
                group_obs = obs_dict[group_name]
                obs_dict[group_name] = torch.cat([group_obs[name] for name in term_names], dim=dim)
        return TensorDict(obs_dict, batch_size=[self.num_envs], device=self.device)

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
