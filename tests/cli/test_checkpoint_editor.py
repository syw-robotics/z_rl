from __future__ import annotations

import torch

from z_rl.cli.checkpoint_editor import build_content_rows, save_checkpoint


def test_build_content_rows_exposes_nested_checkpoint_content() -> None:
    checkpoint = {
        "actor_state_dict": {
            "distribution.std_param": torch.tensor([0.1, 0.2]),
            "head.weight": torch.zeros(2, 3),
        },
        "iter": 4,
    }

    rows = build_content_rows(checkpoint)

    assert ["/actor_state_dict", "dict", "", "dict with 2 entries"] in rows
    assert any(row[0] == "/actor_state_dict/distribution.std_param" for row in rows)
    assert any(row[0] == "/iter" and row[3] == "4" for row in rows)


def test_build_content_rows_filters_nested_checkpoint_content() -> None:
    checkpoint = {
        "actor_state_dict": {
            "distribution.std_param": torch.ones(3),
            "head.weight": torch.zeros(2, 3),
        }
    }

    rows = build_content_rows(checkpoint, filter_text="std_param")

    assert len(rows) == 1
    assert rows[0][0] == "/actor_state_dict/distribution.std_param"


def test_build_content_rows_can_show_all_rows_without_limit() -> None:
    checkpoint = {"items": {f"k{i}": i for i in range(3)}}

    rows = build_content_rows(checkpoint, max_rows=0)

    assert len(rows) == 4
    assert rows[-1] == ["/items/k2", "int", "", "2"]


def test_save_checkpoint_still_renames_top_level_keys_only(tmp_path) -> None:
    checkpoint = {
        "student_state_dict": {
            "distribution.std_param": torch.ones(2),
        },
        "iter": 3,
    }
    output_path = tmp_path / "renamed.pt"

    save_checkpoint(
        checkpoint,
        [
            [True, "student_state_dict", "actor_state_dict"],
            [True, "iter", "iter"],
        ],
        str(output_path),
        "",
    )

    loaded = torch.load(output_path)
    assert "actor_state_dict" in loaded
    assert "student_state_dict" not in loaded
    assert torch.equal(loaded["actor_state_dict"]["distribution.std_param"], torch.ones(2))
    assert loaded["iter"] == 3
