# Z-RL

**Z-RL** is a lightweight reinforcement learning infrastructure derived from [RSL-RL](https://github.com/leggedrobotics/rsl_rl), redesigned for faster iteration in robotics projects.

`Z` is the last letter of the alphabet, we hope Z-RL is your **last** time deeply touching RL infra code.  
Users may implement mixins externally through Z-RL's plugin system, thereby not modifying the source code.

## Key Features

Compared with plain RSL-RL style usage, Z-RL emphasizes:

- **Mixin-first design** for cleaner extension points in algorithms and models.
- **Plugin system** so project-specific logic can live outside the core library.
- **Adaptor layer** for environment integration, currently including IsaacLab `ManagerBasedRLEnv` support.

## Installation

Before installing Z-RL, make sure Python `3.9+` is available.

It is recommended to use a virtual environment (`venv`, `conda`, or `uv`) and activate it first.

### Install from PyPI

```bash
pip install z-rl-lib
```

### Install for development

```bash
git clone https://github.com/syw-robotics/z_rl
cd z_rl
python -m pip install -e .
```

## Usage

For IsaacLab integration details, see:

- `z_rl/adaptor/isaaclab/README.md`
- `z_rl/adaptor/isaaclab/README.zh.md`

## Plugin System

Z-RL supports **external plugin packages** so your custom algorithms/models/modules stay isolated from upstream core code.

### Generate a plugin template

After installing Z-RL, run:

```bash
z-rl-plugin-init

# z-rl-plugin-init --path ./my_zrl_plugin --name z_rl_plugin_example
```

This creates a minimal package scaffold containing:

- custom algorithm mixin example (`MyPPO`)
- custom model mixin examples
- plugin-side IsaacLab config classes (`rl_cfg.py`)

Implement your mixins, then install your plugin in editable mode:

```bash
cd my_zrl_plugin
uv pip install -e .
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
