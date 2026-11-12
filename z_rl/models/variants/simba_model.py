from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from z_rl.models.composition import ComposableModel, HeadSpec
from z_rl.modules import SimBa


@dataclass
class SimBaHeadSpec(HeadSpec):
    """Build a SimBaV2 hyperspherical output head."""

    hidden_dim: int = 512
    num_blocks: int = 2
    expansion: int = 4
    c_shift: float = 3.0

    def build(self, model: nn.Module, input_dim: int, output_dim: int, activation: str) -> nn.Module:
        del model, activation
        if not isinstance(output_dim, int):
            raise ValueError("`SimBaHeadSpec` requires a flat integer output dimension.")
        return SimBa(input_dim, output_dim, self.hidden_dim, self.num_blocks, self.expansion, self.c_shift)


class SimBaModel(ComposableModel):
    """Named preset that installs ``SimBaHeadSpec``."""

    head_spec_class = SimBaHeadSpec
