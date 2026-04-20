from __future__ import annotations

from pathlib import Path

from z_rl.cli.plugin_template import _get_current_zrl_version, create_plugin_template


def test_create_plugin_template_pins_current_zrl_version(tmp_path) -> None:
    root = create_plugin_template(tmp_path / "plugin", "z_rl_plugin_example")
    zrl_version = _get_current_zrl_version()

    pyproject_content = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert f'dependencies = ["z_rl=={zrl_version}"]' in pyproject_content


def test_pyproject_exposes_checkpoint_key_editor_cli() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject_content = pyproject_path.read_text(encoding="utf-8")

    assert 'z-rl-checkpoint-key-editor = "z_rl.cli.checkpoint_key_editor:main"' in pyproject_content
