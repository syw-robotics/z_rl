
from __future__ import annotations

import abc
import torch

from z_rl.storage import RolloutStorage


class BasePPOMixin(abc.ABC):
    """Mixin to extend PPO losses with additional terms.

    Usage:
        1. Inherit from this mixin before :class:`PPO` (e.g. ``class MyAlgo(BasePPOMixin, PPO): ...``).
        2. Override :meth:`compute_additional_loss`.
        3. Return extra losses as dictionaries:
           - ``opt_losses``: included in optimization objective
           - ``non_opt_losses``: logged only

    Notes:
        - Keys in ``opt_losses`` are weighted in ``PPO.update`` by ``<key>_coef`` if such an attribute exists.
        - Extra loss keys must not collide with base PPO loss keys.
    """

    def compute_loss(self, minibatch: RolloutStorage.Batch) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute base PPO losses and merge additional losses from subclasses."""
        base_opt_losses, base_non_opt_losses = super().compute_loss(minibatch)
        extra_opt_losses, extra_non_opt_losses = self.compute_additional_loss(minibatch)

        base_opt_losses.update(extra_opt_losses)
        base_non_opt_losses.update(extra_non_opt_losses)
        return base_opt_losses, base_non_opt_losses

    def compute_additional_loss(
        self, minibatch: RolloutStorage.Batch
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return additional ``(opt_losses, non_opt_losses)``.

        Subclasses can override this to inject custom loss terms on top of PPO.
        """
        return {}, {}
