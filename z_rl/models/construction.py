"""Construction and initialization shared by config-driven models."""

from __future__ import annotations

from tensordict import TensorDict

from z_rl.env import VecEnv
from z_rl.models.mlp_model import MLPModel
from z_rl.utils import inject_obs_time_slice_map, resolve_class


def build_model(
    cfg: dict, obs: TensorDict, groups: dict[str, list[str]], role: str, output_dim: int, env: VecEnv, device: str,
    **model_kwargs,
) -> MLPModel:
    """Build a configured model with the environment's observation metadata."""
    model_class, options = resolve_class(cfg)
    inject_obs_time_slice_map(options, model_class, env)
    options.pop("init_weights", None)
    options.pop("cnn_init_weights", None)
    # An unset spec also works when a composable preset selects a plain model.
    for key in ("latent_spec", "head_spec"):
        if options.get(key) is None:
            options.pop(key, None)
    options.update(model_kwargs)
    return model_class(obs, groups, role, output_dim, **options).to(device)


def init_head_weights(model: MLPModel, cfg: dict, role: str) -> None:
    """Apply the configured orthogonal gain to the head."""
    gain = cfg.get("init_weights")
    if gain is not None:
        model.head.init_weights(gain)
        print(f"{role} Head uses orthogonal init: {gain}")


def init_cnn_weights(model: MLPModel, cfg: dict, role: str) -> None:
    """Apply the encoder's initialization on request."""
    if cfg.get("cnn_init_weights"):
        cnn = getattr(getattr(model, "latent_adapter", None), "cnn", None)
        if not hasattr(cnn, "init_cnn_weights"):
            raise ValueError(f"{role} received cnn_init_weights=True but the model does not define a CNN encoder.")
        cnn.init_cnn_weights()
        print(f"{role} CNNs use kaiming init")
