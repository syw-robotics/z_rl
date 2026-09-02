# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.


from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize values using cumulative or exponentially weighted batch moments.

    Statistics keep the batch axis and may use singleton feature dimensions, allowing values such as ``(C, 1, 1)``
    to normalize inputs shaped ``(..., C, H, W)``.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        until: int | None = None,
        decay: float = 1.0,
        stats_shape: int | tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Initialize the normalizer."""
        super().__init__()
        input_shape = _as_size(shape)
        stats_shape = input_shape if stats_shape is None else _as_size(stats_shape)
        if len(input_shape) != len(stats_shape):
            raise ValueError("stats_shape must have the same rank as shape.")
        if any(stats_dim not in (1, input_dim) for input_dim, stats_dim in zip(input_shape, stats_shape)):
            raise ValueError(f"stats_shape {tuple(stats_shape)} must broadcast to input shape {tuple(input_shape)}.")
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}.")

        self.input_shape = input_shape
        self.stats_shape = stats_shape
        self.decay = decay
        self.eps = eps
        self.until = until
        self._reduction_dims = tuple(
            index + 1
            for index, (input_dim, stats_dim) in enumerate(zip(input_shape, stats_shape))
            if input_dim != stats_dim
        )
        self.register_buffer("_mean", torch.zeros(stats_shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(stats_shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(stats_shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_ema_count", torch.tensor(0.0))

    @property
    def mean(self) -> torch.Tensor:
        """Current running mean."""
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        """Current running standard deviation."""
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mean and variance of values based on empirical values."""
        return (x - self._mean) / (self._std + self.eps)

    @torch.no_grad()
    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Update running moments from a batch while the normalizer is training."""
        if not self.training or (self.until is not None and self.count >= self.until):
            return

        x = x.reshape(-1, *self.input_shape).to(dtype=self._mean.dtype)
        reduction_dims = (0, *self._reduction_dims)
        batch_mean = x.mean(dim=reduction_dims, keepdim=True)
        batch_var = x.var(dim=reduction_dims, unbiased=False, keepdim=True)
        batch_count = x.shape[0]

        # Decay the effective history before merging the new batch. With decay=1 this is the exact cumulative update.
        history_count = self._ema_count * self.decay
        effective_count = history_count + batch_count
        history_weight = history_count / effective_count
        batch_weight = batch_count / effective_count
        mean = history_weight * self._mean + batch_weight * batch_mean

        # Parallel variance merge includes the shift of each population mean relative to the merged mean.
        var = history_weight * (self._var + (self._mean - mean).square()) + batch_weight * (
            batch_var + (batch_mean - mean).square()
        )

        self._mean.copy_(mean)
        self._var.copy_(var)
        self._std.copy_(torch.sqrt(var.clamp_min(0.0)))
        self.count.add_(batch_count)
        self._ema_count.copy_(effective_count)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Checkpoints predating EMA have only the cumulative count, which is also the correct initial effective count.
        ema_count_key = prefix + "_ema_count"
        if ema_count_key not in state_dict:
            count = state_dict.get(prefix + "count", self.count)
            state_dict[ema_count_key] = count.to(dtype=self._ema_count.dtype)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize values based on empirical values."""
        return y * (self._std + self.eps) + self._mean


def _as_size(shape: int | tuple[int, ...] | list[int]) -> torch.Size:
    return torch.Size((shape,)) if isinstance(shape, int) else torch.Size(shape)


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize the scale of the
    rewards so that the value function can learn quickly. We did this by dividing the rewards by a running estimate of
    the standard deviation of the sum of discounted rewards.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        gamma: float = 0.99,
        until: int | None = None,
    ) -> None:
        """Initialize discounted-reward normalization with running moments."""
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using the running std of discounted returns."""
        if self.training:
            # Update discounted rewards
            avg = self.disc_avg.update(rew)
            # Update moments from discounted rewards
            self.emp_norm.update(avg)

        # Normalize rewards with the empirical std
        if self.emp_norm._std > 0:  # type: ignore
            return rew / self.emp_norm._std  # type: ignore
        else:
            return rew


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t
    """

    def __init__(self, gamma: float) -> None:
        """Initialize discounted accumulation with a fixed discount factor."""
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        """Update and return the discounted running average."""
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
