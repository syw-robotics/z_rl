# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for observation selector helper utilities."""

import torch

from z_rl.utils import ObsSelector, resolve_target_obs_term_selector


class TestResolveTargetObsTermSelector:
    """Tests for target observation term selector resolution from cached temporal selectors."""

    def test_resolve_target_obs_term_selector_uses_cached_temporal_selector(self) -> None:
        """The helper should compose the cached last-frame selector with term layout metadata."""
        obs_format = {"policy": {"a": (2, 2), "b": (2, 3)}}
        obs_group_time_slice_map = {
            "policy": {"last": ObsSelector(torch.tensor([2, 3, 7, 8, 9], dtype=torch.long))}
        }
        term_selector = resolve_target_obs_term_selector("policy", ["b"], obs_group_time_slice_map, obs_format)
        obs = torch.arange(10, dtype=torch.float32).reshape(1, -1)

        assert torch.equal(term_selector.meta, torch.tensor([7, 8, 9], dtype=torch.long))
        assert torch.equal(term_selector.select(obs), torch.tensor([[7.0, 8.0, 9.0]]))

    def test_resolve_target_obs_term_selector_preserves_contiguous_slice(self) -> None:
        """Contiguous target terms should stay as a slice when the cached selector is a slice."""
        obs_format = {"policy": {"a": (2, 2), "b": (2, 3), "c": (2, 1)}}
        obs_group_time_slice_map = {"policy": {"last": ObsSelector(slice(10, 16))}}

        term_selector = resolve_target_obs_term_selector("policy", ["b", "c"], obs_group_time_slice_map, obs_format)

        assert term_selector.meta == slice(12, 16)

    def test_resolve_target_obs_term_selector_uses_index_tensor_for_non_contiguous_terms(self) -> None:
        """Non-contiguous target terms should fall back to explicit indices."""
        obs_format = {"policy": {"a": (2, 2), "b": (2, 3), "c": (2, 1)}}
        obs_group_time_slice_map = {"policy": {"last": ObsSelector(slice(10, 16))}}

        term_selector = resolve_target_obs_term_selector("policy", ["a", "c"], obs_group_time_slice_map, obs_format)

        assert torch.equal(term_selector.meta, torch.tensor([10, 11, 15], dtype=torch.long))
