from __future__ import annotations

from z_rl.modules import MoE

from .base_mixin import BaseModelMixin


class MoEHeadMixin(BaseModelMixin):
    """Mixin that replaces the parent model head with a Mixture-of-Experts head."""

    modifies_head = True

    def build_custom_head(self, input_dim: int, output_dim: int, activation: str) -> MoE:
        """Build the MoE head using model attributes configured by the concrete class."""
        return MoE(
            input_dim,
            output_dim,
            self.num_experts,
            self.expert_hidden_dims,
            gate_hidden_dims=self.gate_hidden_dims,
            activation=activation,
        )

    def validate_mixin_contract(self) -> None:
        """Validate MoE-specific initialization contract."""
        if self.num_experts <= 0:
            raise ValueError(f"`num_experts` must be positive, got {self.num_experts}.")
        if isinstance(self.expert_hidden_dims, (tuple, list)) and len(self.expert_hidden_dims) == 0:
            raise ValueError("`expert_hidden_dims` can not be empty.")

    def init_moe_head(self, output_dim: int, activation: str) -> None:
        """Install and initialize the MoE head after the parent model has been constructed."""
        self.validate_mixin_contract()
        head_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.install_custom_head(self.get_latent_dim(), head_output_dim, activation)
        if self.distribution is not None:
            self.head.init_distribution_heads(self.distribution)
