# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for resolve_callable utility function."""

import pytest

from z_rl.utils import resolve_callable


# Test fixtures - nested class for testing nested attribute resolution
class OuterClass:
    """Outer class for testing nested attribute resolution."""

    class InnerClass:
        """Inner nested class."""

        pass

    @staticmethod
    def static_method() -> str:
        """Return a static marker string for callable resolution tests."""
        return "static"


def sample_function() -> str:
    """Sample function for testing."""
    return "sample"


class TestResolveCallableDirect:
    """Tests for direct callable passing."""

    def test_direct_class(self) -> None:
        """Passing a class directly should return it unchanged."""
        from z_rl.algorithms import PPO

        result = resolve_callable(PPO)
        assert result is PPO

    def test_direct_function(self) -> None:
        """Passing a function directly should return it unchanged."""
        result = resolve_callable(sample_function)
        assert result is sample_function

    def test_direct_builtin(self) -> None:
        """Passing a builtin should return it unchanged."""
        result = resolve_callable(len)
        assert result is len


class TestResolveCallableColonFormat:
    """Tests for colon-separated format 'module:attr'."""

    def test_colon_format_class(self) -> None:
        """Should resolve 'module:Class' format."""
        result = resolve_callable("z_rl.algorithms:PPO")
        from z_rl.algorithms import PPO

        assert result is PPO

    def test_colon_format_nested(self) -> None:
        """Should resolve 'module:Outer.Inner' nested format."""
        result = resolve_callable("tests.utils.test_resolve_callable:OuterClass.InnerClass")
        assert result is OuterClass.InnerClass

    def test_colon_format_static_method(self) -> None:
        """Should resolve nested static methods."""
        result = resolve_callable("tests.utils.test_resolve_callable:OuterClass.static_method")
        assert result is OuterClass.static_method

    def test_colon_format_invalid_module(self) -> None:
        """Should raise ImportError for invalid module."""
        with pytest.raises(ImportError):
            resolve_callable("nonexistent_module:SomeClass")

    def test_colon_format_invalid_attr(self) -> None:
        """Should raise AttributeError for invalid attribute."""
        with pytest.raises(AttributeError):
            resolve_callable("z_rl.algorithms:NonexistentClass")


class TestResolveCallableDotFormat:
    """Tests for dot-separated format 'module.attr'."""

    def test_dot_format_class(self) -> None:
        """Should resolve 'module.Class' format."""
        result = resolve_callable("z_rl.algorithms.PPO")
        from z_rl.algorithms import PPO

        assert result is PPO

    def test_dot_format_nested(self) -> None:
        """Should resolve 'module.Outer.Inner' nested format."""
        # This tests the progressive module path splitting
        result = resolve_callable("tests.utils.test_resolve_callable.OuterClass.InnerClass")
        assert result is OuterClass.InnerClass

    def test_dot_format_static_method(self) -> None:
        """Should resolve nested static methods."""
        result = resolve_callable("tests.utils.test_resolve_callable.OuterClass.static_method")
        assert result is OuterClass.static_method

    def test_dot_format_invalid_module(self) -> None:
        """Should raise ImportError for invalid module."""
        with pytest.raises(ImportError):
            resolve_callable("nonexistent_module.SomeClass")

    def test_dot_format_invalid_attr(self) -> None:
        """Should raise AttributeError for invalid attribute."""
        with pytest.raises(AttributeError):
            resolve_callable("z_rl.algorithms.NonexistentClass")


class TestResolveCallableSimpleName:
    """Tests for simple name resolution via z_rl packages."""

    def test_simple_name(self) -> None:
        """Should resolve 'PPO' from z_rl.algorithms."""
        result = resolve_callable("PPO")
        from z_rl.algorithms import PPO

        assert result is PPO

    def test_simple_logger_writer_names(self) -> None:
        """Should resolve optional logger writers without initializing external services."""
        assert resolve_callable("WandbLogWriter").__name__ == "WandbLogWriter"
        assert resolve_callable("NeptuneLogWriter").__name__ == "NeptuneLogWriter"
        assert resolve_callable("SwanLabLogWriter").__name__ == "SwanLabLogWriter"

    def test_simple_name_unknown(self) -> None:
        """Should raise ValueError for unknown simple names."""
        with pytest.raises(ValueError, match="Could not resolve"):
            resolve_callable("NonexistentClassName")


class TestResolveCallableErrors:
    """Tests for error handling."""

    def test_type_error_none(self) -> None:
        """Should raise TypeError for None input."""
        with pytest.raises(TypeError, match="Expected callable or string"):
            resolve_callable(None)  # type: ignore

    def test_type_error_int(self) -> None:
        """Should raise TypeError for int input."""
        with pytest.raises(TypeError, match="Expected callable or string"):
            resolve_callable(42)  # type: ignore

    def test_type_error_list(self) -> None:
        """Should raise TypeError for list input."""
        with pytest.raises(TypeError, match="Expected callable or string"):
            resolve_callable(["PPO"])  # type: ignore


class TestResolveClass:
    """Tests for resolve_class config copying."""

    def test_resolve_class_does_not_mutate_original_cfg(self) -> None:
        """The original config should keep class_name after resolution."""
        from z_rl.utils import resolve_class
        from z_rl.algorithms import PPO

        cfg = {"class_name": "PPO", "learning_rate": 1e-3, "nested": {"std_type": "scalar"}}
        resolved, class_cfg = resolve_class(cfg)

        assert resolved is PPO
        assert "class_name" not in class_cfg
        assert class_cfg["learning_rate"] == 1e-3
        assert cfg["class_name"] == "PPO"
        assert cfg["nested"]["std_type"] == "scalar"
        class_cfg["nested"]["std_type"] = "log"
        assert cfg["nested"]["std_type"] == "scalar"


class TestResolveSpec:
    """Tests for resolve_spec composition references."""

    def test_none_returns_none(self) -> None:
        from z_rl.utils import resolve_spec

        assert resolve_spec(None) is None

    def test_instance_returned_unchanged(self) -> None:
        from z_rl.algorithms import MoERoutingLossSpec
        from z_rl.utils import resolve_spec

        spec = MoERoutingLossSpec(expert_balance_loss_coef=0.2)
        assert resolve_spec(spec) is spec

    def test_class_is_constructed(self) -> None:
        from z_rl.algorithms import MoERoutingLossSpec
        from z_rl.utils import resolve_spec

        spec = resolve_spec(MoERoutingLossSpec)
        assert isinstance(spec, MoERoutingLossSpec)
        assert spec.expert_balance_loss_coef == 1.0e-4

    def test_dict_uses_class_name(self) -> None:
        from z_rl.algorithms import MoERoutingLossSpec
        from z_rl.utils import resolve_spec

        spec = resolve_spec({"class_name": "MoERoutingLossSpec", "gate_entropy_loss_coef": 0.5})
        assert isinstance(spec, MoERoutingLossSpec)
        assert spec.gate_entropy_loss_coef == 0.5

    def test_simple_name_string(self) -> None:
        from z_rl.models import MoEHeadSpec
        from z_rl.utils import resolve_spec

        spec = resolve_spec("MoEHeadSpec")
        assert isinstance(spec, MoEHeadSpec)
        assert spec.num_experts == 4

