# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Generator

import torch


class AmpStorage:
    """Storage for AMP state transitions collected from policy and reference streams.

    AMP compares transitions, not isolated observations:

        policy:    s_t -> s_{t+1}
        reference: s_t -> s_{t+1}

    This storage mirrors the [time, env, feature] layout used by RolloutStorage, but keeps
    only the fields needed by the discriminator.
    """

    class Batch:
        """Mini-batch of AMP policy/reference transitions."""

        def __init__(
            self,
            policy_states: torch.Tensor,
            policy_next_states: torch.Tensor,
            reference_states: torch.Tensor,
            reference_next_states: torch.Tensor,
        ) -> None:
            self.policy_states = policy_states
            self.policy_next_states = policy_next_states
            self.reference_states = reference_states
            self.reference_next_states = reference_next_states

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        amp_obs_shape: tuple[int, ...] | torch.Size,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.amp_obs_shape = tuple(amp_obs_shape)

        storage_shape = (num_transitions_per_env, num_envs, *self.amp_obs_shape)

        with torch.inference_mode(False):
            self.policy_states = torch.zeros(storage_shape, device=device)
            self.policy_next_states = torch.zeros(storage_shape, device=device)
            self.reference_states = torch.zeros(storage_shape, device=device)
            self.reference_next_states = torch.zeros(storage_shape, device=device)
        
        self.step = 0

    def add_transition(
        self,
        policy_state: torch.Tensor,
        policy_next_state: torch.Tensor,
        reference_state: torch.Tensor,
        reference_next_state: torch.Tensor,
    ) -> None:
        """Store one vectorized AMP transition for all environments."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("AMP buffer overflow! You should call clear() before adding new transitions.")

        self.policy_states[self.step].copy_(policy_state)
        self.policy_next_states[self.step].copy_(policy_next_state)
        self.reference_states[self.step].copy_(reference_state)
        self.reference_next_states[self.step].copy_(reference_next_state)
        self.step += 1

    def clear(self) -> None:
        """Reset the write cursor for the next rollout."""
        self.step = 0

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[Batch, None, None]:
        """Yield shuffled flat mini-batches of AMP transitions."""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        policy_states = self.policy_states.flatten(0, 1)
        policy_next_states = self.policy_next_states.flatten(0, 1)
        reference_states = self.reference_states.flatten(0, 1)
        reference_next_states = self.reference_next_states.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                yield AmpStorage.Batch(
                    policy_states=policy_states[batch_idx],
                    policy_next_states=policy_next_states[batch_idx],
                    reference_states=reference_states[batch_idx],
                    reference_next_states=reference_next_states[batch_idx],
                )
