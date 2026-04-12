# Z-RL

**Z-RL** is a lightweight reinforcement learning infrastructure derived from [RSL-RL](https://github.com/leggedrobotics/rsl_rl), redesigned for faster iteration in robotics projects (only supports IsaacLab's manager based rl env currently).

`Z` is the last letter of the alphabet, we hope Z-RL is your **last** time dedicatedly reviewing RL infra code.  

## Key Features

Compared with plain RSL-RL style usage, Z-RL emphasizes:

- **Mixin-first design** for cleaner implementation in algorithms and models.
- **Plugin system** so project-specific logic can live outside the core library.
- **Adaptor layer** for environment integration, currently including IsaacLab `ManagerBasedRLEnv` support.

## Installation

Before installing Z-RL, make sure Python `3.9+` is available.

It is recommended to use a virtual environment (`venv`, `conda`, or `uv`) and activate it first.

```bash
git clone https://github.com/syw-robotics/z_rl
cd z_rl
python -m pip install -e .
```

## Usage

For detailed module guides, see:

- [IsaacLab adaptor README](z_rl/adaptor/isaaclab/README.md)
- [Algorithms README](z_rl/algorithms/README.md)
- [Models README](z_rl/models/README.md)

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
