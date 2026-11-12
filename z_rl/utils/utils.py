# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import importlib
import inspect
import pkgutil
import torch
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from tensordict import TensorDict
from typing import Any, Callable, Literal

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
        if isinstance(self.meta, slice):
            if self.meta.start is None or self.meta.stop is None or self.meta.step not in (None, 1):
                raise ValueError(f"`ObsSelector` only supports explicit contiguous slices, got {self.meta}.")
            object.__setattr__(self, "dim", self.meta.stop - self.meta.start)
            object.__setattr__(self, "_select_impl", self._slice_select)
            return
        if isinstance(self.meta, torch.Tensor):
            object.__setattr__(self, "dim", int(self.meta.numel()))
            object.__setattr__(self, "_select_impl", self._tensor_select)
            return
        raise TypeError(f"`ObsSelector` expects `slice | torch.Tensor`, got {type(self.meta)}.")

    def select(self, obs: torch.Tensor) -> torch.Tensor:
        """Select features from a concatenated observation tensor."""
        return self._select_impl(obs)

    def _slice_select(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, self.meta]  # type: ignore[index]

    def _tensor_select(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.index_select(dim=1, index=self.meta)  # type: ignore[arg-type]


ObsTemporalSelectType = Literal["last", "exclude_last", "exclude_first"]


def resolve_obs_temporal_selector(
    obs_group_name: str,
    temporal_select_type: ObsTemporalSelectType,
    obs_group_time_slice_map: dict[str, dict[str, ObsSelector | slice | torch.Tensor]],
) -> ObsSelector:
    """Resolve a cached temporal selector for one observation group.

    ``temporal_select_type`` is one of ``"last"``, ``"exclude_last"``, ``"exclude_first"``.
    """
    group_selectors = obs_group_time_slice_map.get(obs_group_name, {})
    if temporal_select_type not in group_selectors:
        raise KeyError(
            f"Temporal selector '{temporal_select_type}' for observation group '{obs_group_name}' not found in the"
            " cached `obs_group_time_slice_map`. Available selectors are: "
            f"{list(group_selectors.keys())}"
        )
    selector = group_selectors[temporal_select_type]
    if isinstance(selector, ObsSelector):
        return selector
    return ObsSelector(selector)


def resolve_target_obs_term_selector(
    target_obs_group_name: str,
    target_obs_term_names: Sequence[str],
    obs_group_time_slice_map: dict[str, dict[str, ObsSelector]],
    obs_format: dict[str, dict[str, tuple[int, ...]]],
) -> ObsSelector:
    """Resolve the cached selector metadata for one or more target observation terms."""
    if len(target_obs_term_names) == 0:
        raise ValueError("`target_obs_term_names` can not be empty.")

    last_obs_selector = resolve_obs_temporal_selector(target_obs_group_name, "last", obs_group_time_slice_map)
    group_format = obs_format[target_obs_group_name]

    term_layout: dict[str, tuple[int, int]] = {}
    term_offset = 0
    for term_name, term_format in group_format.items():
        term_dim = int(torch.Size(term_format[1:]).numel())
        term_layout[term_name] = (term_offset, term_offset + term_dim)
        term_offset += term_dim

    ranges: list[tuple[int, int]] = []
    for target_obs_term_name in target_obs_term_names:
        if target_obs_term_name not in term_layout:
            raise KeyError(f"Unknown observation term '{target_obs_term_name}' in group '{target_obs_group_name}'.")
        ranges.append(term_layout[target_obs_term_name])

    start, stop = ranges[0][0], ranges[-1][1]
    is_contiguous = all(prev_stop == curr_start for (_, prev_stop), (curr_start, _) in zip(ranges, ranges[1:]))
    if is_contiguous:
        if isinstance(last_obs_selector.meta, slice):
            return ObsSelector(slice(last_obs_selector.meta.start + start, last_obs_selector.meta.start + stop))
        return ObsSelector(last_obs_selector.meta[start:stop])

    if isinstance(last_obs_selector.meta, slice):
        base = last_obs_selector.meta.start
        indices = [idx for term_start, term_stop in ranges for idx in range(base + term_start, base + term_stop)]
        return ObsSelector(torch.tensor(indices, dtype=torch.long))
    return ObsSelector(torch.cat([last_obs_selector.meta[term_start:term_stop] for term_start, term_stop in ranges]))


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

    for set_name, groups in obs_groups.items():
        if len(groups) == 0:
            raise ValueError(f"The '{set_name}' key in the 'obs_groups' dictionary can not be an empty list.")
        for group in groups:
            if group not in obs:
                raise ValueError(
                    f"Observation '{group}' in observation set '{set_name}' not found in the observations from the"
                    f" environment. Available observations from the environment: {list(obs.keys())}"
                )

    for default_set_name in default_sets:
        if default_set_name in obs_groups:
            continue
        if default_set_name in obs:
            obs_groups[default_set_name] = [default_set_name]
            warnings.warn(
                f"The observation configuration dictionary 'obs_groups' does not contain the '{default_set_name}'"
                f" key. As an observation group with the name '{default_set_name}' was found, this is assumed to be"
                f" the appropriate observation. Consider adding the '{default_set_name}' key to the 'obs_groups'"
                f" dictionary for clarity. This behavior will be removed in a future version."
            )
            continue
        if "policy" in obs:
            obs_groups[default_set_name] = ["policy"]
            warnings.warn(
                f"The observation configuration dictionary 'obs_groups' does not contain the '{default_set_name}'"
                f" key. As an observation group with the name 'policy' was found, this is assumed to be the"
                f" appropriate observation. Consider adding the '{default_set_name}' key to the 'obs_groups'"
                f" dictionary for clarity. This behavior will be removed in a future version."
            )
            continue
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
    """Inject env observation metadata into model config when the constructor accepts it."""
    init_params = inspect.signature(model_class.__init__).parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in init_params.values())
    if hasattr(env, "obs_group_time_slice_map") and ("obs_group_time_slice_map" in init_params or accepts_kwargs):
        model_cfg.setdefault("obs_group_time_slice_map", env.obs_group_time_slice_map)
    if hasattr(env, "obs_format") and ("obs_format" in init_params or accepts_kwargs):
        model_cfg.setdefault("obs_format", env.obs_format)


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


def compile_model(model: torch.nn.Module, mode: str | None = None) -> torch.nn.Module:
    """Compile a model when requested, rejecting CUDA-graph modes that break PPO-style multi-model calls."""
    if mode is None:
        return model
    if mode in ("reduce-overhead", "max-autotune"):
        raise ValueError(
            f"torch_compile_mode='{mode}' uses CUDA graphs and is incompatible with the algorithm forward pattern. "
            "Use 'default', 'max-autotune-no-cudagraphs', or None."
        )
    return torch.compile(model, mode=mode)  # type: ignore


def reduce_gradients_in_buckets(params: Iterable[torch.nn.Parameter], world_size: int, bucket_mb: float) -> None:
    """Average gradients across GPUs in bounded-size buckets.

    Gradients are packed into buffers of at most ``bucket_mb`` and reduced with a single
    ``all_reduce`` call per buffer. A gradient larger than the bucket on its own is reduced in
    contiguous slices instead. This bounds the size of the temporary packed buffer while still
    batching small gradients together to limit the number of collective calls.

    Args:
        params: Parameters whose gradients should be reduced. Parameters with no gradient are skipped.
        world_size: Number of distributed processes to average the summed gradients over.
        bucket_mb: Maximum size, in megabytes, of a single packed buffer.
    """
    bucket_bytes = int(bucket_mb * 1024 * 1024)
    grads = [param.grad.view(-1) for param in params if param.grad is not None]
    start = 0
    while start < len(grads):
        nbytes = grads[start].numel() * grads[start].element_size()
        if nbytes > bucket_bytes:
            chunk_numel = max(1, bucket_bytes // grads[start].element_size())
            for offset in range(0, grads[start].numel(), chunk_numel):
                chunk = grads[start].narrow(0, offset, min(chunk_numel, grads[start].numel() - offset))
                torch.distributed.all_reduce(chunk, op=torch.distributed.ReduceOp.SUM)
                chunk /= world_size
            start += 1
            continue

        filled_bytes = 0
        end = start
        while end < len(grads):
            grad_bytes = grads[end].numel() * grads[end].element_size()
            if filled_bytes + grad_bytes > bucket_bytes:
                break
            filled_bytes += grad_bytes
            end += 1

        packed = torch.cat(grads[start:end])
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        packed /= world_size
        offset = 0
        for flat_grad in grads[start:end]:
            numel = flat_grad.numel()
            flat_grad.copy_(packed[offset : offset + numel])
            offset += numel
        start = end


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
        padded = {key: _pad_time_major_trajectories(value, trajectory_lengths_list) for key, value in tensor.items()}
        padded_trajectories = TensorDict(
            padded, batch_size=[tensor.batch_size[0], len(trajectory_lengths_list)], device=tensor.device
        )
    else:
        padded_trajectories = _pad_time_major_trajectories(tensor, trajectory_lengths_list)

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def _pad_time_major_trajectories(tensor: torch.Tensor, trajectory_lengths: list[int]) -> torch.Tensor:
    """Split a ``[T, N, ...]`` tensor at done boundaries and pad to the longest trajectory."""
    pieces = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths)
    pad_row = torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device)
    return torch.nn.utils.rnn.pad_sequence((*pieces, pad_row))[:, :-1]  # type: ignore[index]


def unpad_trajectories(trajectories: torch.Tensor | TensorDict, masks: torch.Tensor) -> torch.Tensor | TensorDict:
    """Do the inverse operation of `split_and_pad_trajectories()`."""
    valid_steps = trajectories.transpose(1, 0)[masks.transpose(1, 0)]
    if isinstance(trajectories, TensorDict):
        # TensorDict.view() only modifies the batch size.
        return valid_steps.view(-1, trajectories.shape[0]).transpose(1, 0)
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
    if act_name not in act_dict:
        raise ValueError(f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}")
    return act_dict[act_name]


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
    if optimizer_name not in optimizer_dict:
        raise ValueError(f"Invalid optimizer '{optimizer_name}'. Valid optimizers are: {list(optimizer_dict.keys())}")
    return optimizer_dict[optimizer_name]


def resolve_class(cfg: dict) -> tuple[Callable, dict]:
    """Resolve the class referenced by ``cfg["class_name"]`` without mutating ``cfg``.

    Args:
        cfg: Configuration dictionary with a ``"class_name"`` key and the constructor arguments of that class.

    Returns:
        The resolved class and a deep copy of ``cfg`` with ``"class_name"`` removed.
    """
    class_cfg = copy.deepcopy(cfg)
    return resolve_callable(class_cfg.pop("class_name")), class_cfg


def resolve_spec(spec_ref: Any) -> Any:
    """Resolve a composition spec from an instance, class, class name, or config dict.

    Accepted references:

    - instance: returned unchanged
    - class: constructed with ``cls()``
    - string: resolved with :func:`resolve_callable`, then constructed
    - dict: resolved with :func:`resolve_class`, then constructed with the remaining kwargs

    Args:
        spec_ref: Spec instance, class, import path, or ``{"class_name": ..., ...}`` dict.

    Returns:
        A spec instance, or ``None`` when ``spec_ref`` is ``None``.
    """
    if spec_ref is None:
        return None
    if not isinstance(spec_ref, (dict, str, type)):
        return spec_ref
    if isinstance(spec_ref, dict):
        spec_cls, cfg = resolve_class(spec_ref)
    elif isinstance(spec_ref, str):
        spec_cls, cfg = resolve_callable(spec_ref), {}
    else:
        spec_cls, cfg = spec_ref, {}
    return spec_cls(**cfg)


def spec_init_field_names(spec: object) -> set[str]:
    """Return settable constructor field names for a spec instance."""
    if is_dataclass(spec):
        return {item.name for item in fields(spec) if item.init}
    try:
        return {name for name in vars(spec) if not name.startswith("_")}
    except TypeError:
        return set()


def bind_matching_fields(spec: object, cfg: dict, *, exclude: Iterable[str] = ()) -> None:
    """Move leftover config keys that match spec init fields onto the spec.

    Named presets keep a flat constructor/config. ``MLPEncoderModel(..., encoder_latent_dim=128)`` and IsaacLab
    ``ZRlMLPEncoderModelCfg.encoder_latent_dim`` are not ``MLPModel`` arguments; this copies them onto the latent
    spec before ``MLPModel.__init__`` sees the kwargs.

    The same flattening is used for algorithm configs: ``estimation_loss_coef`` sits next to PPO
    hyperparameters, and ``ComposablePPO.build_loss_spec`` binds it onto the loss spec.
    """
    names = spec_init_field_names(spec) - set(exclude)
    for key in list(cfg):
        if key in names:
            setattr(spec, key, cfg.pop(key))


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
        obj: Any = importlib.import_module(module_path)
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj

    if "." in callable_or_name:
        return _resolve_dotted_callable(callable_or_name)

    for _, module_name, _ in pkgutil.iter_modules(z_rl.__path__, "z_rl."):
        module = importlib.import_module(module_name)
        if hasattr(module, callable_or_name):
            return getattr(module, callable_or_name)
    raise ValueError(
        f"Could not resolve '{callable_or_name}'. Use qualified name like 'module.path:ClassName' "
        f"or pass the class directly."
    )


def _resolve_dotted_callable(name: str) -> Callable:
    """Resolve ``module.path.Class.Nested`` by trying every valid module/attr split."""
    parts = name.split(".")
    module_found = False
    for i in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        module_found = True
        obj: Any = module
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    if module_found:
        raise AttributeError(f"Could not resolve '{name}': attribute not found in module")
    raise ImportError(f"Could not resolve '{name}': no valid module.attr split found")
