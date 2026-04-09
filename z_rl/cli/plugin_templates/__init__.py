"""Template snippets for z_rl plugin generator."""

from .algorithms import ALGORITHMS_INIT_TEMPLATE, ALGORITHMS_MY_PPO_TEMPLATE
from .models import MODELS_INIT_TEMPLATE, MODELS_MODEL_TEMPLATE
from .project import render_readme_template, render_rl_cfg_template

__all__ = [
    "ALGORITHMS_INIT_TEMPLATE",
    "ALGORITHMS_MY_PPO_TEMPLATE",
    "MODELS_INIT_TEMPLATE",
    "MODELS_MODEL_TEMPLATE",
    "render_readme_template",
    "render_rl_cfg_template",
]
