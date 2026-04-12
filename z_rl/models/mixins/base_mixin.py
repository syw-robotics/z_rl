from __future__ import annotations

import abc
import torch.nn as nn


class BaseModelMixin(abc.ABC):
    """Hook-oriented base class for model mixins.

    This base mixin intentionally stays thin and only defines extension points around latent-adapter installation and
    head customization. Concrete model classes remain responsible for observation parsing, latent construction, forward
    execution, and export wrappers.

    Latent adapters are expected to be exportable ``nn.Module`` instances that accept the normalized latent. Mixins
    that change the latent dimensionality should expose it through :meth:`get_latent_dim` so the concrete model can
    initialize its head consistently.
    """

    modifies_latent: bool = False
    """Whether the mixin replaces the parent model latent adapter."""

    modifies_head: bool = False
    """Whether the mixin replaces the parent model head."""

    def build_latent_adapter(self) -> nn.Module | None:
        """Build an optional replacement latent adapter.

        Returning ``None`` keeps the parent model latent adapter unchanged. Mixins that change the adapter output
        dimensionality must coordinate that with the eventual head input dimension.
        """
        return None

    def install_latent_adapter(self) -> nn.Module | None:
        """Build and install an optional replacement latent adapter."""
        latent_adapter = self.build_latent_adapter()
        if latent_adapter is not None:
            self.latent_adapter = latent_adapter
        return latent_adapter

    def get_latent_dim(self) -> int | None:
        """Return the latent dimensionality produced after the adapter.

        Returning ``None`` means the mixin does not override the parent model's latent size.
        """
        return None

    def build_custom_head(self, input_dim: int, output_dim: int, activation: str) -> nn.Module | None:
        """Build an optional replacement head.

        Returning ``None`` keeps the parent model head unchanged.
        """
        return None

    def install_custom_head(self, input_dim: int, output_dim: int, activation: str) -> nn.Module | None:
        """Build and install an optional replacement head."""
        head = self.build_custom_head(input_dim, output_dim, activation)
        if head is not None:
            self.head = head
        return head

    def validate_mixin_contract(self) -> None:
        """Validate mixin-specific assumptions after the parent model has been initialized."""
        return None
