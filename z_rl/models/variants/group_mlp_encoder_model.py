from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
from tensordict import TensorDict

from z_rl.models.composition import ComposableModel, GroupObsLatentAdapter, LatentSpec
from z_rl.modules import EmpiricalNormalization, MLP
from z_rl.utils import resolve_obs_temporal_selector


@dataclass
class GroupMLPLatentSpec(LatentSpec):
    """Encode each active 1D observation group with its own MLP."""

    encoder_cfgs: dict[str, dict[str, Any]] = field(default_factory=dict)
    append_last_obs: bool = False
    append_obs_group: str = "policy"
    _latent_dim: int = field(init=False, repr=False, default=0)

    def validate(self, model: nn.Module) -> None:
        expected = set(model.obs_groups)
        configured = set(self.encoder_cfgs)
        if configured != expected:
            raise ValueError(
                "`GroupMLPLatentSpec.encoder_cfgs` must declare every active observation group, "
                f"got {sorted(configured)} vs obs_groups={sorted(expected)}."
            )
        for group in model.obs_groups:
            if len(model.obs_group_shapes[group]) != 1:
                raise ValueError(
                    f"`GroupMLPLatentSpec` only supports 1D observations, got "
                    f"shape {model.obs_group_shapes[group]} for '{group}'."
                )
            if "output_dim" not in self.encoder_cfgs[group]:
                raise ValueError(f"Encoder config for '{group}' requires `output_dim`.")
        if self.append_last_obs and self.append_obs_group not in model.obs_groups:
            raise ValueError(
                f"`GroupMLPLatentSpec.append_last_obs` requires a '{self.append_obs_group}' observation group."
            )

    def build(self, model: nn.Module) -> nn.Module:
        encoders = {}
        normalizers = {}
        group_dims = []
        self._latent_dim = 0

        def _normalization_kwargs(value: bool | dict[str, Any]) -> dict[str, Any] | None:
            """Map a normalization switch to ``EmpiricalNormalization`` kwargs.

            ``False`` disables normalization. ``True`` uses the module defaults. A dictionary is forwarded as kwargs.
            """
            if value is False:
                return None
            if value is True:
                return {}
            return value

        default_normalization_cfg = _normalization_kwargs(model.obs_normalization)

        for group in model.obs_groups:
            input_dim = int(model.obs_group_shapes[group][-1])
            cfg = dict(self.encoder_cfgs[group])
            if "obs_normalization" in cfg:
                normalization_cfg = _normalization_kwargs(cfg.pop("obs_normalization"))
            else:
                normalization_cfg = default_normalization_cfg
            output_dim = int(cfg.pop("output_dim"))
            hidden_dims = cfg.pop("hidden_dims", ())
            activation = cfg.pop("activation", "swish")
            layer_norm = cfg.pop("layer_norm", None)
            if cfg:
                raise ValueError(f"Unsupported encoder configuration keys for '{group}': {list(cfg)}")
            if hidden_dims:
                encoder = MLP(
                    input_dim,
                    output_dim,
                    hidden_dims,
                    activation,
                    layer_norm=layer_norm,
                    first_non_muon=True,
                    last_non_muon=True,
                )
            else:
                encoder = nn.Linear(input_dim, output_dim)
                encoder.weight._non_muon = True
            encoders[group] = encoder
            normalizers[group] = (
                nn.Identity()
                if normalization_cfg is None
                else EmpiricalNormalization(input_dim, **normalization_cfg)
            )
            group_dims.append(input_dim)
            self._latent_dim += output_dim

        append_obs = None
        if self.append_last_obs:
            append_obs = resolve_obs_temporal_selector(
                self.append_obs_group, "last", model.obs_group_time_slice_map
            )
        return GroupObsLatentAdapter(
            obs_groups=list(model.obs_groups),
            obs_group_dims=group_dims,
            encoders=encoders,
            obs_normalizers=normalizers,
            append_obs=append_obs,
            append_obs_group=self.append_obs_group,
        )

    def get_latent_dim(self, model: nn.Module) -> int:
        if not self.append_last_obs:
            return self._latent_dim
        return self._latent_dim + resolve_obs_temporal_selector(
            self.append_obs_group, "last", model.obs_group_time_slice_map
        ).dim


class GroupMLPEncoderModel(ComposableModel):
    """Named preset that installs ``GroupMLPLatentSpec``."""

    latent_spec_class = GroupMLPLatentSpec
