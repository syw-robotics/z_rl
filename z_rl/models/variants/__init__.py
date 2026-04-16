"""Predefined model variants."""

from .encoder_mlp_model import EncoderMLPModel
from .moe_model import MoEModel
from .vae_model import VAEModel

__all__ = ["EncoderMLPModel", "MoEModel", "VAEModel"]
