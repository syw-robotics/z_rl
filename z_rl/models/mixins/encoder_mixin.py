from __future__ import annotations

import abc
import copy
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from z_rl.models.mlp_model import _OnnxMLPModel, _TorchMLPModel
from z_rl.models.mixins.model_mixin import BaseModelMixin
from z_rl.modules import HiddenState, MLP
from z_rl.utils import get_obs_time_slice


class BaseEncoderMixin(BaseModelMixin, abc.ABC):
    """Base mixin for policy-only latent encoders."""

    def __init__(
        self,
        *args: Any,
        encoder_output_dim: int = 256,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        encoder_activation: str = "elu",
        concat_policy_last_obs: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(*args, **kwargs)
        if encoder_output_dim <= 0:
            raise ValueError(f"`encoder_output_dim` must be positive, got {encoder_output_dim}.")
        if not (isinstance(self.obs_groups, list) and len(self.obs_groups) == 1 and self.obs_groups[0] == "policy"):
            raise ValueError("Encoder mixins require exactly one active observation group named 'policy'.")

        self.encoder_output_dim = encoder_output_dim
        self.encoder_obs_dim = self.obs_dim
        self.concat_policy_last_obs = concat_policy_last_obs

        self.policy_last_obs_dim = 0
        self.policy_last_obs_selector: slice | torch.Tensor | None = None
        if self.concat_policy_last_obs:
            time_slice_map = self.obs_group_time_slice_map if isinstance(self.obs_group_time_slice_map, dict) else {}
            if "policy" not in time_slice_map or "last" not in time_slice_map["policy"]:
                raise ValueError(
                    "`concat_policy_last_obs=True` requires `obs_group_time_slice_map['policy']['last']` to exist."
                )
            selector = time_slice_map["policy"]["last"]
            self.policy_last_obs_selector = selector
            self.policy_last_obs_dim = int(selector.numel())

        self.encoder = self._build_encoder_module(encoder_hidden_dims, encoder_activation)

    @abc.abstractmethod
    def _build_encoder_module(
        self, encoder_hidden_dims: tuple[int, ...] | list[int], encoder_activation: str
    ) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def _encode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        latent = super().get_latent(obs, masks, hidden_state)
        encoded_latent = self._encode_latent(latent)

        if not self.concat_policy_last_obs:
            return encoded_latent

        policy_last_obs = get_obs_time_slice(obs["policy"], "policy", "last", self.obs_group_time_slice_map)
        return torch.cat([encoded_latent, policy_last_obs], dim=-1)

    def _get_latent_dim(self) -> int:
        return self.encoder_output_dim + (self.policy_last_obs_dim if self.concat_policy_last_obs else 0)

    @abc.abstractmethod
    def as_jit(self) -> nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError


class MLPEncoderMixin(BaseEncoderMixin):
    """Mixin that encodes policy latent with an MLP encoder."""

    def _build_encoder_module(
        self, encoder_hidden_dims: tuple[int, ...] | list[int], encoder_activation: str
    ) -> torch.nn.Module:
        return MLP(self.encoder_obs_dim, self.encoder_output_dim, encoder_hidden_dims, encoder_activation)

    def _encode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.encoder(latent)

    def as_jit(self) -> nn.Module:
        return _TorchMLPEncodedMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxMLPEncodedMLPModel(self, verbose)


class _TorchMLPEncodedMLPModel(_TorchMLPModel):
    """Exportable encoded MLP model for JIT (MLP encoder)."""

    def __init__(self, model: BaseEncoderMixin) -> None:
        super().__init__(model)  # type: ignore[arg-type]
        self.encoder = copy.deepcopy(model.encoder)
        self.policy_obs_dim = model.encoder_obs_dim
        self.concat_policy_last_obs = model.concat_policy_last_obs
        self.policy_last_obs_selector = (
            copy.deepcopy(model.policy_last_obs_selector) if self.concat_policy_last_obs else None
        )

    def _build_output_latent(self, raw_x: torch.Tensor, normalized_x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(normalized_x[..., : self.policy_obs_dim])
        if not self.concat_policy_last_obs:
            return encoded

        if isinstance(self.policy_last_obs_selector, slice):
            policy_last_obs = raw_x[..., : self.policy_obs_dim][..., self.policy_last_obs_selector]
        else:
            policy_last_obs = raw_x[..., : self.policy_obs_dim].index_select(
                dim=-1, index=self.policy_last_obs_selector.to(device=raw_x.device)
            )
        return torch.cat([encoded, policy_last_obs], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_x = x
        normalized_x = self.obs_normalizer(x)
        latent = self._build_output_latent(raw_x, normalized_x)
        out = self.head(latent)
        return self.deterministic_output(out)


class _OnnxMLPEncodedMLPModel(_OnnxMLPModel):
    """Exportable encoded MLP model for ONNX (MLP encoder)."""

    def __init__(self, model: BaseEncoderMixin, verbose: bool) -> None:
        super().__init__(model, verbose)  # type: ignore[arg-type]
        self.encoder = copy.deepcopy(model.encoder)
        self.policy_obs_dim = model.encoder_obs_dim
        self.concat_policy_last_obs = model.concat_policy_last_obs
        self.policy_last_obs_selector = (
            copy.deepcopy(model.policy_last_obs_selector) if self.concat_policy_last_obs else None
        )
        self.input_size = model.obs_dim

    def _build_output_latent(self, raw_x: torch.Tensor, normalized_x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(normalized_x[..., : self.policy_obs_dim])
        if not self.concat_policy_last_obs:
            return encoded

        if isinstance(self.policy_last_obs_selector, slice):
            policy_last_obs = raw_x[..., : self.policy_obs_dim][..., self.policy_last_obs_selector]
        else:
            policy_last_obs = raw_x[..., : self.policy_obs_dim].index_select(
                dim=-1, index=self.policy_last_obs_selector.to(device=raw_x.device)
            )
        return torch.cat([encoded, policy_last_obs], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_x = x
        normalized_x = self.obs_normalizer(x)
        latent = self._build_output_latent(raw_x, normalized_x)
        out = self.head(latent)
        return self.deterministic_output(out)
