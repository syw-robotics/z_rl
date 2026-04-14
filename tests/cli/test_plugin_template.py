from __future__ import annotations

from z_rl.cli.plugin_template import _get_current_zrl_version, create_plugin_template


def test_create_plugin_template_pins_current_zrl_version(tmp_path) -> None:
    root = create_plugin_template(tmp_path / "plugin", "z_rl_plugin_example")
    zrl_version = _get_current_zrl_version()

    pyproject_content = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert f'dependencies = ["z_rl=={zrl_version}"]' in pyproject_content
