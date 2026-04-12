from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn

from z_rl.models.mixins.base_mixin import BaseModelMixin
from z_rl.modules import MLP


class BaseEncoderMixin(BaseModelMixin, abc.ABC):
    """Base mixin for policy-only latent encoders.

    This mixin changes the latent dimensionality, so it computes the encoder output shape before the parent model
    initializes its head, exposes it through :meth:`get_latent_dim`, and installs the latent adapter
    immediately after parent initialization.
    """

    modifies_latent = True

    def __init__(
        self,
        *args: Any,
        encoder_output_dim: int = 256,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        encoder_activation: str = "elu",
        concat_policy_last_obs: bool = False,
        **kwargs: Any,
    ) -> None:
        self.encoder_output_dim = encoder_output_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.encoder_activation = encoder_activation
        self.concat_policy_last_obs = concat_policy_last_obs
        self.encoder_obs_dim = self._infer_policy_obs_dim(*args, **kwargs)

        self.policy_last_obs_dim = 0
        self.policy_last_obs_selector: slice | torch.Tensor | None = None
        if self.concat_policy_last_obs:
            time_slice_map = kwargs.get("obs_group_time_slice_map")
            if "policy" in time_slice_map and "last" in time_slice_map["policy"]:
                selector = time_slice_map["policy"]["last"]
                self.policy_last_obs_selector = selector
                if isinstance(selector, slice):
                    start, stop, step = selector.indices(self.encoder_obs_dim)
                    self.policy_last_obs_dim = len(range(start, stop, step))
                else:
                    self.policy_last_obs_dim = int(selector.numel())

        super().__init__(*args, **kwargs)

        self.validate_mixin_contract()
        self.install_latent_adapter()

    @abc.abstractmethod
    def build_latent_adapter(self) -> nn.Module:
        """Build and return the encoder-backed latent adapter."""
        raise NotImplementedError

    def validate_mixin_contract(self) -> None:
        """Validate encoder-specific initialization contract."""
        if self.encoder_output_dim <= 0:
            raise ValueError(f"`encoder_output_dim` must be positive, got {self.encoder_output_dim}.")
        if not (isinstance(self.obs_groups, list) and len(self.obs_groups) == 1 and self.obs_groups[0] == "policy"):
            raise ValueError("Encoder mixins require exactly one active observation group named 'policy'.")
        if self.concat_policy_last_obs:
            time_slice_map = self.obs_group_time_slice_map if isinstance(self.obs_group_time_slice_map, dict) else {}
            if "policy" not in time_slice_map or "last" not in time_slice_map["policy"]:
                raise ValueError(
                    "`concat_policy_last_obs=True` requires `obs_group_time_slice_map['policy']['last']` to exist."
                )

    @staticmethod
    def _infer_policy_obs_dim(*args: Any, **kwargs: Any) -> int:
        """Infer the flattened policy observation dimension from constructor inputs."""
        obs = kwargs.get("obs", args[0] if len(args) > 0 else None)
        if obs is None or "policy" not in obs:
            return 0
        return int(obs["policy"].shape[-1])

    @property
    def encoder(self) -> nn.Module:
        """Return the encoder submodule owned by the latent adapter.

        Custom encoder adapters are expected to expose an ``encoder`` attribute.
        """
        return self.latent_adapter.encoder  # type: ignore[return-value]

    def get_latent_dim(self) -> int:
        return self.encoder_output_dim + (self.policy_last_obs_dim if self.concat_policy_last_obs else 0)


class MLPEncoderMixin(BaseEncoderMixin):
    """Mixin that encodes policy latent with an MLP encoder."""

    def build_latent_adapter(self) -> nn.Module:
        return _MLPEncoderLatentAdapter(
            input_dim=self.encoder_obs_dim,
            output_dim=self.encoder_output_dim,
            hidden_dims=self.encoder_hidden_dims,
            activation=self.encoder_activation,
            concat_policy_last_obs=self.concat_policy_last_obs,
            policy_last_obs_selector=self.policy_last_obs_selector,
        )


class _MLPEncoderLatentAdapter(nn.Module):
    """Exportable latent adapter that encodes normalized policy observations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        concat_policy_last_obs: bool,
        policy_last_obs_selector: slice | torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.encoder = MLP(input_dim, output_dim, hidden_dims, activation)
        self.policy_obs_dim = input_dim
        self.concat_policy_last_obs = concat_policy_last_obs
        self.register_buffer(
            "policy_last_obs_indices", self._selector_to_indices(policy_last_obs_selector, input_dim), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the normalized policy latent and optionally append the last frame."""
        encoded = self.encoder(x[..., : self.policy_obs_dim])
        if not self.concat_policy_last_obs:
            return encoded

        policy = x[..., : self.policy_obs_dim]
        policy_last_obs = policy.index_select(dim=-1, index=self.policy_last_obs_indices.to(device=x.device))
        return torch.cat([encoded, policy_last_obs], dim=-1)

    @staticmethod
    def _selector_to_indices(selector: slice | torch.Tensor | None, input_dim: int) -> torch.Tensor:
        """Convert a last-axis selector into explicit indices for export-safe indexing."""
        if selector is None:
            return torch.zeros(0, dtype=torch.long)
        if isinstance(selector, slice):
            start, stop, step = selector.indices(input_dim)
            return torch.arange(start, stop, step, dtype=torch.long)
        return selector.to(dtype=torch.long)


