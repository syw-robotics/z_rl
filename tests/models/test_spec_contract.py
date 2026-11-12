from __future__ import annotations

import pytest
import torch.nn as nn

from z_rl.models.composition import HeadSpec, LatentSpec


class _IdentityAdapter(nn.Module):
    def forward(self, x):
        return x


class ValidLatentSpec(LatentSpec):
    def build(self, model: nn.Module) -> nn.Module:
        return _IdentityAdapter()

    def get_latent_dim(self, model: nn.Module) -> int:
        del model
        return 8


class ValidHeadSpec(HeadSpec):
    def build(self, model: nn.Module, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        return nn.Linear(input_dim, output_dim)


def test_latent_spec_contract_accepts_required_overrides() -> None:
    spec = ValidLatentSpec()

    assert isinstance(spec.build(nn.Identity()), nn.Module)
    assert spec.get_latent_dim(nn.Identity()) == 8


def test_head_spec_contract_accepts_required_override() -> None:
    spec = ValidHeadSpec()

    assert isinstance(spec.build(nn.Identity(), 8, 4, "elu"), nn.Module)


def test_latent_spec_requires_build() -> None:
    class MissingAdapterSpec(LatentSpec):
        def get_latent_dim(self, model: nn.Module) -> int:
            del model
            return 8

    with pytest.raises(TypeError, match="abstract method"):
        MissingAdapterSpec()


def test_latent_spec_requires_get_latent_dim() -> None:
    class MissingDimSpec(LatentSpec):
        def build(self, model: nn.Module) -> nn.Module:
            return _IdentityAdapter()

    with pytest.raises(TypeError, match="abstract method"):
        MissingDimSpec()


def test_head_spec_requires_build() -> None:
    class MissingHeadBuilderSpec(HeadSpec):
        pass

    with pytest.raises(TypeError, match="abstract method"):
        MissingHeadBuilderSpec()


def test_head_spec_validate_is_optional() -> None:
    spec = ValidHeadSpec()
    spec.validate(nn.Identity())
