"""Project-level template renderers for plugin scaffold generation."""

from __future__ import annotations


def render_readme_template(project_name: str, package_name: str) -> str:
    """Render README.md content for a generated plugin project."""
    return f"""# {project_name}

Plugin template for `z_rl`.

## Install

```bash
uv pip install -e .
```

## Structure

```text
{package_name}/
├── algorithms
├── models
├── modules
└── rl_cfg.py
```

The generated algorithm template follows the current `ComposablePPO` pattern:

- define a `PPOLossSpec`
- subclass `ComposablePPO`
- override `build_loss_spec(env, algorithm_cfg)`
"""


def render_rl_cfg_template(package_name: str) -> str:
    """Render rl_cfg.py content for generated plugin package."""
    return f"""from __future__ import annotations

from isaaclab.utils import configclass

from z_rl.adaptor.isaaclab.rl_cfg import ZRlMLPModelCfg, ZRlPpoAlgorithmCfg


@configclass
class MyPpoAlgorithmCfg(ZRlPpoAlgorithmCfg):
    class_name: str = "{package_name}.algorithms.my_ppo:MyPPO"
    my_aux_loss_coef: float = 0.1


@configclass
class MyActorModelCfg(ZRlMLPModelCfg):
    class_name: str = "{package_name}.models.my_model:MyActorModel"


@configclass
class MyCriticModelCfg(ZRlMLPModelCfg):
    class_name: str = "{package_name}.models.my_model:MyCriticModel"
"""
