"""Project-level template renderers for plugin scaffold generation."""

from __future__ import annotations


def render_readme_template(project_name: str, package_name: str) -> str:
    """Render README.md content for a generated plugin project."""
    return f"""# {project_name}

Plugin template for `z_rl`.

## Install

```bash
python -m pip install -e .
```

## Structure

```text
{package_name}/
├── algorithms
├── models
├── modules
└── rl_cfg.py
```

The generated templates follow spec-first composition:

- implement a `PPOLossSpec` / `LatentSpec` / `HeadSpec`
- point IsaacLab config at `ComposablePPO` / `ComposableModel` plus the spec
"""


def render_rl_cfg_template(package_name: str) -> str:
    """Render rl_cfg.py content for generated plugin package."""
    return f"""from __future__ import annotations

from isaaclab.utils import configclass

from z_rl.adaptor.isaaclab.rl_cfg import ZRlComposableModelCfg, ZRlComposablePpoAlgorithmCfg


@configclass
class MyPpoAlgorithmCfg(ZRlComposablePpoAlgorithmCfg):
    loss_spec: dict = {{
        "class_name": "{package_name}.algorithms.my_loss:MyAuxLossSpec",
        "my_aux_loss_coef": 0.1,
    }}


@configclass
class MyActorModelCfg(ZRlComposableModelCfg):
    latent_spec: dict = {{"class_name": "{package_name}.models.my_model:MyLatentSpec"}}
    head_spec: dict = {{"class_name": "{package_name}.models.my_model:MyHeadSpec"}}


@configclass
class MyCriticModelCfg(ZRlComposableModelCfg):
    latent_spec: dict = {{"class_name": "{package_name}.models.my_model:MyLatentSpec"}}
"""
