# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import importlib
import inspect
import pkgutil
import torch
import warnings
from dataclasses import dataclass, field
from tensordict import TensorDict
from typing import Any, Callable

import z_rl


"""
Observation opration utilities
"""


@dataclass(frozen=True, slots=True)
class ObsSelector:
    """Resolved observation selector with pre-dispatched dim/select operations."""

    meta: slice | torch.Tensor
    dim: int = 0
    _select_impl: Callable[[torch.Tensor], torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.meta, slice):  # if meta type is slice
            if self.meta.start is None or self.meta.stop is None or self.meta.step not in (None, 1):
                raise ValueError(f"`ObsSelector` only supports explicit contiguous slices, got {self.meta}.")
            object.__setattr__(self, "dim", self.meta.stop - self.meta.start)
            object.__setattr__(self, "_select_impl", self._slice_select)
        elif isinstance(self.meta, torch.Tensor):  # if meta type is tensor
            object.__setattr__(self, "dim", int(self.meta.numel()))
            object.__setattr__(self, "_select_impl", self._tensor_select)
        else:
            raise TypeError(f"`ObsSelector` expects `slice | torch.Tensor`, got {type(self.meta)}.")

    def select(self, obs: torch.Tensor) -> torch.Tensor:
        """Select features from a concatenated observation tensor."""
        return self._select_impl(obs)

    def _slice_select(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, self.meta]  # type: ignore[index]

    def _tensor_select(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.index_select(dim=1, index=self.meta)  # type: ignore[arg-type]


def resolve_obs_temporal_selector(
    obs_group_name: str,
    temporal_select_type: str,
    obs_group_time_slice_map: dict[str, dict[str, ObsSelector]],
) -> ObsSelector:
    """Resolve the cached selector metadata."""
    if temporal_select_type not in obs_group_time_slice_map.get(obs_group_name, {}):
        raise KeyError(
            f"Temporal selector '{temporal_select_type}' for observation group '{obs_group_name}' not found in the"
            " cached `obs_group_time_slice_map`. Available selectors are: "
            f"{list(obs_group_time_slice_map.get(obs_group_name, {}).keys())}"
        )
    return obs_group_time_slice_map[obs_group_name][temporal_select_type]

def resolve_target_obs_term_selector(
    target_obs_group_name: str,
    target_obs_term_names: list[str],
    obs_group_time_slice_map: dict[str, dict[str, ObsSelector]],
    obs_format: dict[str, dict[str, tuple[int, ...]]],
) -> ObsSelector:
    """Resolve the cached selector metadata for one or more target observation terms."""
    last_obs_selector = resolve_obs_temporal_selector(target_obs_group_name, "last", obs_group_time_slice_map)
    if len(target_obs_term_names) == 0:
        raise ValueError("`target_obs_term_names` can not be empty.")
    group_format = obs_format[target_obs_group_name]

    # Build the single-frame layout once: each term maps to its [start, stop) range.
    term_layout: dict[str, tuple[int, int]] = {}
    term_offset = 0
    for term_name, term_format in group_format.items():
        term_dim = int(torch.Size(term_format[1:]).numel())
        term_layout[term_name] = (term_offset, term_offset + term_dim)
        term_offset += term_dim

    ranges = []
    for target_obs_term_name in target_obs_term_names:
        if target_obs_term_name not in term_layout:
            raise KeyError(f"Unknown observation term '{target_obs_term_name}' in group '{target_obs_group_name}'.")
        ranges.append(term_layout[target_obs_term_name])

    if len(ranges) == 1:
        start, stop = ranges[0]
        if isinstance(last_obs_selector.meta, slice):
            return ObsSelector(slice(last_obs_selector.meta.start + start, last_obs_selector.meta.start + stop))
        return ObsSelector(last_obs_selector.meta[start:stop])

    # Keep contiguous terms as a single slice / view. Fall back to explicit indices only when needed.
    is_contiguous = all(prev_stop == curr_start for (_, prev_stop), (curr_start, _) in zip(ranges, ranges[1:]))
    start = ranges[0][0]
    stop = ranges[-1][1]
    if is_contiguous:
        if isinstance(last_obs_selector.meta, slice):
            return ObsSelector(slice(last_obs_selector.meta.start + start, last_obs_selector.meta.start + stop))
        return ObsSelector(last_obs_selector.meta[start:stop])

    if isinstance(last_obs_selector.meta, slice):
        indices = [
            feature_idx
            for start, stop in ranges
            for feature_idx in range(last_obs_selector.meta.start + start, last_obs_selector.meta.start + stop)
        ]
        return ObsSelector(torch.tensor(indices, dtype=torch.long))

    return ObsSelector(torch.cat([last_obs_selector.meta[start:stop] for start, stop in ranges]))

def resolve_obs_groups(
    obs: TensorDict, obs_groups: dict[str, list[str]], default_sets: list[str]
) -> dict[str, list[str]]:
    """Validate the observation configuration and resolve missing observation sets.

    The input is an observation dictionary `obs` containing observation groups and a configuration dictionary
    `obs_groups` where the keys are the observation sets and the values are lists of observation groups.

    The configuration dictionary could for example look like::

        {
            "actor": ["group_1", "group_2"],
            "critic": ["group_1", "group_3"],
        }

    This means that the 'actor' observation set will contain the observations "group_1" and "group_2" and the 'critic'
    observation set will contain the observations "group_1" and "group_3". This function will check that all the
    observations in the 'actor' and 'critic' observation sets are present in the observation dictionary from the
    environment.

    Additionally, if one of the `default_sets`, e.g. "critic", is not present in the configuration dictionary, this
    function will:

    1. Check if a group with the same name exists in the observations and assign this group to the observation set.
    2. If 1. fails, it will assign the 'policy' observation group to the missing observation set.
    3. If 2. fails, an error is raised.

    Args:
        obs: Observations from the environment in the form of a dictionary.
        obs_groups: Dictionary mapping observation sets to lists of observation groups.
        default_sets: Default observation set names used by the algorithm. If not provided in ``obs_groups``, a
            default behavior gets triggered.

    Returns:
        The resolved observation groups.

    Raises:
        ValueError: If any observation set is an empty list.
        ValueError: If any observation set contains an observation term that is not present in the observations.
        ValueError: If a default observation set cannot be resolved according to the rules above.
    """
    if len(obs_groups) == 0:
        warnings.warn(
            "The observation configuration dictionary 'obs_groups' is empty and thus likely not configured. Consider"
            " configuring the 'obs_groups' dictionary explicitly"
        )
    else:
        # Check all observation sets for valid observation groups
        for set_name, groups in obs_groups.items():
            # Check if the list is empty
            if len(groups) == 0:
                raise ValueError(f"The '{set_name}' key in the 'obs_groups' dictionary can not be an empty list.")
            # Check groups exist inside the observations from the environment
            for group in groups:
                if group not in obs:
                    raise ValueError(
                        f"Observation '{group}' in observation set '{set_name}' not found in the observations from the"
                        f" environment. Available observations from the environment: {list(obs.keys())}"
                    )

    for default_set_name in default_sets:
        if default_set_name not in obs_groups:
            if default_set_name in obs:
                obs_groups[default_set_name] = [default_set_name]
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' does not contain the '{default_set_name}'"
                    f" key. As an observation group with the name '{default_set_name}' was found, this is assumed to be"
                    f" the appropriate observation. Consider adding the '{default_set_name}' key to the 'obs_groups'"
                    f" dictionary for clarity. This behavior will be removed in a future version."
                )
            elif "policy" in obs:
                obs_groups[default_set_name] = ["policy"]
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' does not contain the '{default_set_name}'"
                    f" key. As an observation group with the name 'policy' was found, this is assumed to be the"
                    f" appropriate observation. Consider adding the '{default_set_name}' key to the 'obs_groups'"
                    f" dictionary for clarity. This behavior will be removed in a future version."
                )
            else:
                raise ValueError(
                    f"The observation configuration dictionary 'obs_groups' does not contain the '{default_set_name}'"
                    f" key and no suitable observation could be found in the observations from the environment."
                    f" Please refer to `z_rl.utils.resolve_obs_groups()` for information on how to configure the"
                    f" 'obs_groups' dictionary correctly."
                )

    print("-" * 80)
    print("Resolved observation sets: ")
    for set_name, groups in obs_groups.items():
        print("\t", set_name, ": ", groups)
    print("-" * 80)

    return obs_groups

def inject_obs_time_slice_map(model_cfg: dict, model_class: type, env: Any) -> None:
    """Inject ``obs_group_time_slice_map`` into model config when supported by the model constructor."""
    if not hasattr(env, "obs_group_time_slice_map"):
        return

    init_params = inspect.signature(model_class.__init__).parameters
    accepts_time_slice_map = "obs_group_time_slice_map" in init_params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in init_params.values()
    )
    if accepts_time_slice_map:
        model_cfg.setdefault("obs_group_time_slice_map", env.obs_group_time_slice_map)


"""
Algorithm utilities
"""


def get_param(param: Any, idx: int) -> Any:
    """Get a parameter for the given index.

    Args:
        param: Parameter or list/tuple of parameters.
        idx: Index to get the parameter for.
    """
    if isinstance(param, (tuple, list)):
        return param[idx]
    else:
        return param

def check_nan(obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor) -> None:
    """Raise ``ValueError`` if any environment output contains NaN."""
    for key, tensor in obs.items():
        if torch.isnan(tensor).any():
            raise ValueError(
                f"The observation group '{key}' returned by the environment contains NaN values. This usually indicates"
                " a bug in the environment's step() or reset() function."
            )
    if torch.isnan(rewards).any():
        raise ValueError(
            "The rewards returned by the environment contain NaN values. This usually indicates a bug in the"
            " environment's reward computation."
        )
    if torch.isnan(dones).any():
        raise ValueError(
            "The dones returned by the environment contain NaN values. This usually indicates a bug in the"
            " environment's termination logic."
        )

def split_and_pad_trajectories(
    tensor: torch.Tensor | TensorDict, dones: torch.Tensor
) -> tuple[torch.Tensor | TensorDict, torch.Tensor]:
    """Split trajectories at done indices.

    Split trajectories, concatenate them and pad with zeros up to the length of the longest trajectory. Return masks
    corresponding to valid parts of the trajectories.

    Example (transposed for readability):
        Input: [[a1, a2, a3, a4 | a5, a6],
                [b1, b2 | b3, b4, b5 | b6]]

        Output:[[a1, a2, a3, a4], | [[True, True, True, True],
                [a5, a6, 0, 0],   |  [True, True, False, False],
                [b1, b2, 0, 0],   |  [True, True, False, False],
                [b3, b4, b5, 0],  |  [True, True, True, False],
                [b6, 0, 0, 0]]    |  [True, False, False, False]]

    Assumes that the input has the following order of dimensions: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    if isinstance(tensor, TensorDict):
        padded_trajectories = {}
        for k, v in tensor.items():
            # Split the tensor into trajectories
            trajectories = torch.split(v.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
            # Add at least one full length trajectory
            trajectories = (*trajectories, torch.zeros(v.shape[0], *v.shape[2:], device=v.device))
            # Pad the trajectories to the length of the longest trajectory
            padded_trajectories[k] = torch.nn.utils.rnn.pad_sequence(trajectories)  # type: ignore
            # Remove the added trajectory
            padded_trajectories[k] = padded_trajectories[k][:, :-1]
        padded_trajectories = TensorDict(
            padded_trajectories, batch_size=[tensor.batch_size[0], len(trajectory_lengths_list)], device=tensor.device
        )
    else:
        # Split the tensor into trajectories
        trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
        # Add at least one full length trajectory
        trajectories = (*trajectories, torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device))
        # Pad the trajectories to the length of the longest trajectory
        padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)  # type: ignore
        # Remove the added trajectory
        padded_trajectories = padded_trajectories[:, :-1]
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories: torch.Tensor | TensorDict, masks: torch.Tensor) -> torch.Tensor | TensorDict:
    """Do the inverse operation of `split_and_pad_trajectories()`."""
    valid_steps = trajectories.transpose(1, 0)[masks.transpose(1, 0)]
    if isinstance(trajectories, TensorDict):
        # TensorDict.view() only modifies the batch size.
        # We reshape [valid_steps] -> [number of envs, time] and then transpose back to [time, number of envs]
        return valid_steps.view(-1, trajectories.shape[0]).transpose(1, 0)
    else:
        # For standard Tensors, we must explicitly handle feature dimensions in view()
        return valid_steps.view(-1, trajectories.shape[0], *trajectories.shape[2:]).transpose(1, 0)

def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """Resolve the activation function from the name.

    Valid activation function names are: ``"elu"``, ``"selu"``, ``"relu"``, ``"crelu"``, ``"lrelu"``, ``"tanh"``,
    ``"sigmoid"``, ``"softplus"``, ``"gelu"``, ``"swish"``, ``"mish"``, ``"identity"``.

    Args:
        act_name: Name of the activation function.

    Returns:
        The activation function.

    Raises:
        ValueError: If the activation function is not found.
    """
    act_dict = {
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "relu": torch.nn.ReLU(),
        "crelu": torch.nn.CELU(),
        "lrelu": torch.nn.LeakyReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "softplus": torch.nn.Softplus(),
        "gelu": torch.nn.GELU(),
        "swish": torch.nn.SiLU(),
        "mish": torch.nn.Mish(),
        "identity": torch.nn.Identity(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}")

def resolve_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
    """Resolve the optimizer from the name.

    Valid optimizer names are: ``"adam"``, ``"adamw"``, ``"sgd"``, ``"rmsprop"``.

    Args:
        optimizer_name: Name of the optimizer.

    Returns:
        The optimizer.

    Raises:
        ValueError: If the optimizer is not found.
    """
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name in optimizer_dict:
        return optimizer_dict[optimizer_name]
    else:
        raise ValueError(f"Invalid optimizer '{optimizer_name}'. Valid optimizers are: {list(optimizer_dict.keys())}")

def resolve_callable(callable_or_name: type | Callable | str) -> Callable:
    """Resolve a callable from a string, type, or return callable directly.

    This function supports resolving callables from a direct callable input or from a string in one of these formats:

    - Direct callable: pass a type or function directly (for example, ``MyClass`` or ``my_func``).
    - Qualified name with colon: ``"module.path:Attr.Nested"`` (explicit, recommended).
    - Qualified name with dot: ``"module.path.ClassName"`` (implicit).
    - Simple name: for example ``"PPO"`` or ``"ActorCritic"`` (searched within ``z_rl``).

    Args:
        callable_or_name: A callable (type/function) or string name.

    Returns:
        The resolved callable.

    Raises:
        TypeError: If input is neither a callable nor a string.
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute cannot be found in the module.
        ValueError: If a simple name cannot be found in z_rl packages.
    """
    if callable(callable_or_name):
        return callable_or_name

    if not isinstance(callable_or_name, str):
        raise TypeError(f"Expected callable or string, got {type(callable_or_name)}")

    if ":" in callable_or_name:
        module_path, attr_path = callable_or_name.rsplit(":", 1)
        # Try to import the module
        module = importlib.import_module(module_path)
        # Try to get the attribute
        obj = module
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj  # type: ignore

    if "." in callable_or_name:
        parts = callable_or_name.split(".")
        module_found = False
        for i in range(len(parts) - 1, 0, -1):
            # Try to import the module with the first i parts
            module_path = ".".join(parts[:i])
            attr_parts = parts[i:]
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError:
                continue
            module_found = True
            # Once a module is found, try to get the attribute
            obj = module
            try:
                for attr in attr_parts:
                    obj = getattr(obj, attr)
                return obj  # type: ignore
            except AttributeError:
                continue
        if module_found:
            raise AttributeError(f"Could not resolve '{callable_or_name}': attribute not found in module")
        else:
            raise ImportError(f"Could not resolve '{callable_or_name}': no valid module.attr split found")

    for _, module_name, _ in pkgutil.iter_modules(z_rl.__path__, "z_rl."):
        module = importlib.import_module(module_name)
        if hasattr(module, callable_or_name):
            return getattr(module, callable_or_name)

    raise ValueError(
        f"Could not resolve '{callable_or_name}'. Use qualified name like 'module.path:ClassName' "
        f"or pass the class directly."
    )
