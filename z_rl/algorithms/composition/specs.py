from __future__ import annotations

import abc

import torch

from z_rl.storage import RolloutStorage


class PPOLossSpec(abc.ABC):
    """Abstract base class for extending PPO with additional optimization or logging terms.

    Required override: ``compute``. ``validate`` is a no-op by default. Spec fields named ``<loss_key>_coef``
    are copied onto the algorithm so ``PPO.update()`` can weight the corresponding loss.
    """

    def validate(self, algo: object) -> None:
        """Validate spec-specific assumptions against the initialized algorithm."""
        del algo

    @abc.abstractmethod
    def compute(
        self,
        algo: object,
        minibatch: RolloutStorage.Batch,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return additional ``(opt_losses, non_opt_losses)`` to merge into the PPO loss dicts."""
        raise NotImplementedError
