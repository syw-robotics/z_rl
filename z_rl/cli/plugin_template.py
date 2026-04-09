from __future__ import annotations

import argparse
import re
from pathlib import Path

from .plugin_templates import (
    ALGORITHMS_INIT_TEMPLATE,
    ALGORITHMS_MY_PPO_TEMPLATE,
    MODELS_INIT_TEMPLATE,
    MODELS_MODEL_TEMPLATE,
    render_readme_template,
    render_rl_cfg_template,
)


def _validate_package_name(name: str) -> str:
    """Validate a Python package name used for generated plugin sources."""
    normalized = name.strip()
    if not normalized:
        raise ValueError("Plugin package name must not be empty.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
        raise ValueError(
            f"Invalid plugin package name '{name}'. Use a valid Python identifier such as 'z_rl_plugin_example'."
        )
    return normalized


def _project_name_from_package(package_name: str) -> str:
    """Return a distribution name derived from the package name."""
    return package_name.replace("_", "-")


def _write_file(path: Path, content: str) -> None:
    """Write a text file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_plugin_template(target_dir: str | Path, package_name: str) -> Path:
    """Create a plugin template project and return the project root path."""
    root = Path(target_dir).expanduser().resolve()
    package_name = _validate_package_name(package_name)

    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Target directory '{root}' already exists and is not empty.")
    root.mkdir(parents=True, exist_ok=True)

    project_name = _project_name_from_package(package_name)
    package_dir = root / package_name

    _write_file(
        root / "pyproject.toml",
        f"""[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "External plugin package for z_rl"
readme = "README.md"
requires-python = ">=3.9"
dependencies = []

[tool.setuptools.packages.find]
where = ["."]
include = ["{package_name}*"]
""",
    )
    _write_file(
        root / "README.md",
        render_readme_template(project_name, package_name),
    )
    _write_file(package_dir / "__init__.py", '"""External z_rl plugin package."""\n')
    _write_file(
        package_dir / "algorithms" / "__init__.py",
        ALGORITHMS_INIT_TEMPLATE,
    )
    _write_file(
        package_dir / "algorithms" / "my_ppo.py",
        ALGORITHMS_MY_PPO_TEMPLATE,
    )
    _write_file(
        package_dir / "models" / "__init__.py",
        MODELS_INIT_TEMPLATE,
    )
    _write_file(
        package_dir / "models" / "my_model.py",
        MODELS_MODEL_TEMPLATE,
    )
    _write_file(package_dir / "modules" / "__init__.py", '"""Custom reusable plugin modules live here."""\n')
    _write_file(
        package_dir / "rl_cfg.py",
        render_rl_cfg_template(package_name),
    )

    return root


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Generate a z_rl plugin template.")
    parser.add_argument("--path", help="Target directory for the generated plugin project.")
    parser.add_argument("--name", help="Python package name for the generated plugin.")
    return parser


def main() -> int:
    """Run the interactive plugin template generator."""
    parser = _build_parser()
    args = parser.parse_args()

    target_dir = args.path or input("Plugin project path: ").strip()
    package_name = args.name or input("Plugin package name [z_rl_plugin_example]: ").strip() or "z_rl_plugin_example"

    try:
        root = create_plugin_template(target_dir, package_name)
    except (FileExistsError, ValueError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    print(f"Created plugin template at: {root}")
    print("Next step: cd into the project and run `uv pip install -e .`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
