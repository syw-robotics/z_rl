from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch.nn as nn


MODULE_PATH = Path(__file__).resolve().parents[2] / "z_rl/models/mixins/base_mixin.py"
SPEC = importlib.util.spec_from_file_location("z_rl_base_mixin_for_test", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
BASE_MIXIN_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE_MIXIN_MODULE)
BaseHeadMixin = BASE_MIXIN_MODULE.BaseHeadMixin
BaseLatentMixin = BASE_MIXIN_MODULE.BaseLatentMixin


class _IdentityAdapter(nn.Module):
    def forward(self, x):
        return x


class ValidLatentMixin(BaseLatentMixin):

    def build_latent_adapter(self) -> nn.Module:
        return _IdentityAdapter()

    def validate_mixin_contract(self) -> None:
        super().validate_mixin_contract()

    def get_latent_dim(self) -> int:
        return 8


class MissingLatentDimMixin(BaseLatentMixin):

    def build_latent_adapter(self) -> nn.Module:
        return _IdentityAdapter()

    def validate_mixin_contract(self) -> None:
        super().validate_mixin_contract()


class InvalidLatentDimMixin(BaseLatentMixin):

    def build_latent_adapter(self) -> nn.Module:
        return _IdentityAdapter()

    def validate_mixin_contract(self) -> None:
        super().validate_mixin_contract()

    def get_latent_dim(self) -> int:
        return 0


class MissingHeadBuilderMixin(BaseHeadMixin):
    def validate_mixin_contract(self) -> None:
        super().validate_mixin_contract()


class ValidHeadMixin(BaseHeadMixin):

    def build_custom_head(self, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        return nn.Linear(input_dim, output_dim)

    def validate_mixin_contract(self) -> None:
        super().validate_mixin_contract()


def test_latent_mixin_contract_accepts_required_overrides() -> None:
    ValidLatentMixin().validate_mixin_contract()


def test_latent_mixin_contract_allows_default_latent_dim() -> None:
    MissingLatentDimMixin().validate_mixin_contract()


def test_latent_mixin_contract_requires_positive_latent_dim() -> None:
    with pytest.raises(NotImplementedError, match="positive integer"):
        InvalidLatentDimMixin().validate_mixin_contract()


def test_head_mixin_contract_requires_build_custom_head_override() -> None:
    with pytest.raises(NotImplementedError, match="build_custom_head"):
        MissingHeadBuilderMixin().validate_mixin_contract()


def test_head_mixin_contract_accepts_required_override() -> None:
    ValidHeadMixin().validate_mixin_contract()
