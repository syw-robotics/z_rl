from __future__ import annotations

import pytest
import torch.nn as nn

from z_rl.models.composition import HeadSpec, LatentSpec


class _IdentityAdapter(nn.Module):
    def forward(self, x):
        return x


class ValidLatentSpec(LatentSpec):
    def validate(self, model: nn.Module) -> None:
        return None

    def build_latent_adapter(self, model: nn.Module) -> nn.Module:
        return _IdentityAdapter()

    def get_latent_dim(self, model: nn.Module) -> int:
        return 8


class ValidHeadSpec(HeadSpec):
    def validate(self, model: nn.Module) -> None:
        return None

    def build_head(self, model: nn.Module, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        return nn.Linear(input_dim, output_dim)


def test_latent_spec_contract_accepts_required_overrides() -> None:
    spec = ValidLatentSpec()

    assert isinstance(spec.build_latent_adapter(nn.Identity()), nn.Module)
    assert spec.get_latent_dim(nn.Identity()) == 8


def test_head_spec_contract_accepts_required_override() -> None:
    spec = ValidHeadSpec()

    assert isinstance(spec.build_head(nn.Identity(), 8, 4, "elu"), nn.Module)


def test_latent_spec_requires_all_abstract_methods() -> None:
    class MissingLatentDimSpec(LatentSpec):
        def validate(self, model: nn.Module) -> None:
            return None

        def build_latent_adapter(self, model: nn.Module) -> nn.Module:
            return _IdentityAdapter()

    with pytest.raises(TypeError, match="abstract method"):
        MissingLatentDimSpec()


def test_head_spec_requires_build_head_override() -> None:
    class MissingHeadBuilderSpec(HeadSpec):
        def validate(self, model: nn.Module) -> None:
            return None

    with pytest.raises(TypeError, match="abstract method"):
        MissingHeadBuilderSpec()
