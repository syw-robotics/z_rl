"""Gradio GUI for inspecting and editing top-level keys in a PyTorch checkpoint."""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

try:
    import gradio as gr
except ModuleNotFoundError:
    gr = None


DEFAULT_CONTENT_MAX_ROWS = 5000
TENSOR_VALUE_PREVIEW_LIMIT = 16


def resolve_input_path(path_text: str, uploaded_file: Any) -> Path:
    if path_text and path_text.strip():
        return Path(path_text.strip()).expanduser().resolve()
    if uploaded_file is not None:
        uploaded_name = getattr(uploaded_file, "name", None)
        if uploaded_name:
            return Path(uploaded_name).expanduser().resolve()
    raise ValueError("Please provide a checkpoint path or upload a checkpoint file.")


def load_checkpoint(source_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(source_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dict checkpoint, got {type(checkpoint).__name__}: {source_path}")
    return checkpoint


def summarize_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        summary = f"tensor shape={tuple(value.shape)} dtype={value.dtype}"
        if value.numel() <= TENSOR_VALUE_PREVIEW_LIMIT:
            summary += f" value={value.detach().cpu().tolist()}"
        return summary
    if isinstance(value, dict):
        return f"dict with {len(value)} entries"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} with {len(value)} items"
    if value is None:
        return "None"
    text = repr(value)
    return text if len(text) <= 80 else f"{text[:77]}..."


def is_parameter_dict(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return any(isinstance(item, torch.Tensor) for item in value.values())


def build_overview_rows(checkpoint: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key, value in checkpoint.items():
        rows.append([key, type(value).__name__, summarize_value(value)])
    return rows


def build_rename_rows(checkpoint: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key in checkpoint:
        rows.append([True, key, key])
    return rows


def suggest_output_path(source_path: Path | None) -> str:
    if source_path is None:
        return ""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return str(source_path.with_name(f"{source_path.stem}_renamed_{timestamp}{source_path.suffix}"))


def choose_default_preview_key(checkpoint: dict[str, Any]) -> str | None:
    for preferred in ("student_state_dict", "actor_state_dict"):
        if preferred in checkpoint and is_parameter_dict(checkpoint[preferred]):
            return preferred
    for key, value in checkpoint.items():
        if is_parameter_dict(value):
            return key
    return None


def build_state_dict_rows(checkpoint: dict[str, Any], selected_key: str | None, max_rows: int = 200) -> list[list[Any]]:
    if not selected_key or selected_key not in checkpoint:
        return []

    value = checkpoint[selected_key]
    if not isinstance(value, dict):
        return []

    rows: list[list[Any]] = []
    for idx, (key, item) in enumerate(value.items()):
        if idx >= max_rows:
            rows.append(["...", "...", "...", f"showing first {max_rows} rows"])
            break
        if isinstance(item, torch.Tensor):
            rows.append([key, str(tuple(item.shape)), str(item.dtype), str(int(item.numel()))])
        else:
            rows.append([key, "-", type(item).__name__, summarize_value(item)])
    return rows


def escape_path_token(token: Any) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def join_path(parent_path: str, key: Any) -> str:
    return f"{parent_path}/{escape_path_token(key)}"


def normalize_max_rows(max_rows: int | float | None) -> int | None:
    if max_rows is None:
        return DEFAULT_CONTENT_MAX_ROWS
    try:
        row_limit = int(max_rows)
    except (TypeError, ValueError):
        return DEFAULT_CONTENT_MAX_ROWS
    if row_limit <= 0:
        return None
    return row_limit


def build_content_rows(
    checkpoint: dict[str, Any],
    filter_text: str = "",
    max_rows: int | float | None = DEFAULT_CONTENT_MAX_ROWS,
) -> list[list[Any]]:
    needle = filter_text.strip().lower() if filter_text else ""
    row_limit = normalize_max_rows(max_rows)

    rows: list[list[Any]] = []
    truncated = False
    stack: list[tuple[str, Any]] = [("", checkpoint)]
    seen_container_ids: set[int] = set()

    while stack:
        path, value = stack.pop()
        if path:
            type_name = type(value).__name__
            shape = str(tuple(value.shape)) if isinstance(value, torch.Tensor) else ""
            summary = summarize_value(value)
            haystack = f"{path} {type_name} {shape} {summary}".lower()
            if not needle or needle in haystack:
                if row_limit is not None and len(rows) >= row_limit:
                    truncated = True
                    break
                rows.append([path, type_name, shape, summary])

        if isinstance(value, torch.Tensor):
            continue
        if isinstance(value, dict):
            container_id = id(value)
            if container_id in seen_container_ids:
                continue
            seen_container_ids.add(container_id)
            for key, item in reversed(list(value.items())):
                stack.append((join_path(path, key), item))
        elif isinstance(value, (list, tuple)):
            container_id = id(value)
            if container_id in seen_container_ids:
                continue
            seen_container_ids.add(container_id)
            for idx in reversed(range(len(value))):
                stack.append((join_path(path, idx), value[idx]))

    if truncated:
        rows.append(["", "", "", f"showing first {row_limit} matching rows; use the path filter to narrow results"])
    return rows


def load_for_ui(path_text: str, uploaded_file: Any):
    source_path = resolve_input_path(path_text, uploaded_file)
    checkpoint = load_checkpoint(source_path)

    preview_key = choose_default_preview_key(checkpoint)
    preview_choices = [key for key, value in checkpoint.items() if isinstance(value, dict)]
    if preview_key is None and preview_choices:
        preview_key = preview_choices[0]

    info = f"Loaded `{source_path}`\n\n- Top-level keys: {len(checkpoint)}"
    if preview_key:
        info += f"\n- Preview state_dict: `{preview_key}`"

    return (
        checkpoint,
        str(source_path),
        gr.update(value=info),
        build_overview_rows(checkpoint),
        build_rename_rows(checkpoint),
        gr.update(choices=preview_choices, value=preview_key),
        build_state_dict_rows(checkpoint, preview_key),
        build_content_rows(checkpoint),
        suggest_output_path(source_path),
    )


def update_preview(checkpoint: dict[str, Any] | None, selected_key: str):
    if not checkpoint:
        return []
    return build_state_dict_rows(checkpoint, selected_key)


def update_content_rows(checkpoint: dict[str, Any] | None, filter_text: str, max_rows: int | float | None):
    if not checkpoint:
        return []
    return build_content_rows(checkpoint, filter_text, max_rows)


def save_checkpoint(
    checkpoint: dict[str, Any] | None,
    rename_rows: Any,
    output_path_text: str,
    loaded_source_path: str,
):
    if not checkpoint:
        raise ValueError("Please load a checkpoint before saving.")

    rows = rename_rows or []
    converted: dict[str, Any] = {}

    for row in rows:
        if len(row) < 3:
            continue
        include, input_key, output_key = row[:3]
        if not include:
            continue
        if not input_key or input_key not in checkpoint:
            raise KeyError(f"Input key not found in checkpoint: {input_key}")
        if not output_key:
            raise ValueError(f"Output key cannot be empty for input key: {input_key}")
        converted[str(output_key)] = checkpoint[str(input_key)]

    if not converted:
        raise ValueError("No checkpoint entries selected for saving.")

    if output_path_text and output_path_text.strip():
        output_path = Path(output_path_text.strip()).expanduser().resolve()
    elif loaded_source_path:
        output_path = Path(suggest_output_path(Path(loaded_source_path).expanduser().resolve()))
    else:
        temp_dir = Path(tempfile.gettempdir())
        output_path = temp_dir / f"renamed_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    message = f"Saved checkpoint to `{output_path}` with {len(converted)} top-level keys."
    return message, str(output_path)


def build_app() -> gr.Blocks:
    if gr is None:
        raise RuntimeError("gradio is not installed. Install it with: python3 -m pip install gradio")

    with gr.Blocks(title="Checkpoint Key Editor") as demo:
        checkpoint_state = gr.State(None)
        source_path_state = gr.State("")

        gr.Markdown(
            """
            <p align="center">
                <h2 align="center">Checkpoint Key Editor
                </h2>
            </p>

            <br>

            > Load a PyTorch checkpoint, inspect all nested contents with path filtering, edit top-level key mappings, and save an updated checkpoint.
            >
            > By default, all top-level keys keep their original names until you change them in the rename plan.
            >
            > The suggested output path uses the same directory as the loaded checkpoint and appends `_renamed_<timestamp>`.

            <br>

            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                path_input = gr.Textbox(
                    label="Checkpoint Path",
                    placeholder="/path/to/checkpoint.pt",
                )
            with gr.Column(scale=2):
                upload_input = gr.File(label="Or Upload Checkpoint", file_count="single")
            with gr.Column(scale=1):
                load_button = gr.Button("Load", variant="primary")

        load_info = gr.Markdown("No checkpoint loaded.")

        with gr.Row():
            with gr.Column(scale=5):
                overview_table = gr.Dataframe(
                    headers=["key", "type", "summary"],
                    datatype=["str", "str", "str"],
                    type="array",
                    interactive=False,
                    label="Top-level Checkpoint Overview",
                    row_count=(0, "dynamic"),
                    wrap=True,
                )
            with gr.Column(scale=5):
                rename_table = gr.Dataframe(
                    headers=["include", "input_key", "output_key"],
                    datatype=["bool", "str", "str"],
                    type="array",
                    interactive=True,
                    label="Top-level Rename Plan",
                    row_count=(0, "dynamic"),
                    wrap=True,
                )

        preview_key = gr.Dropdown(
            label="Preview Nested Dict",
            choices=[],
            value=None,
            allow_custom_value=False,
        )
        state_dict_table = gr.Dataframe(
            headers=["param_key", "shape", "dtype", "numel_or_summary"],
            datatype=["str", "str", "str", "str"],
            type="array",
            interactive=False,
            label="Nested Dict Preview",
            row_count=(0, "dynamic"),
            wrap=True,
        )

        with gr.Accordion("Full Checkpoint Content", open=True):
            with gr.Row():
                with gr.Column(scale=5):
                    content_filter = gr.Textbox(
                        label="Path Filter",
                        placeholder="std_param, log_std_param, optimizer, infos, ...",
                    )
                with gr.Column(scale=2):
                    content_max_rows = gr.Number(
                        label="Max Rows (0 = all)",
                        value=DEFAULT_CONTENT_MAX_ROWS,
                        precision=0,
                    )
                with gr.Column(scale=1):
                    content_filter_button = gr.Button("Filter")
            content_table = gr.Dataframe(
                headers=["path", "type", "shape", "summary"],
                datatype=["str", "str", "str", "str"],
                type="array",
                interactive=False,
                label="Read-only Content View",
                row_count=(0, "dynamic"),
                wrap=True,
            )

        with gr.Row():
            with gr.Column(scale=5):
                output_path = gr.Textbox(
                    label="Output Path",
                    placeholder="/path/to/renamed_checkpoint.pt",
                )
            with gr.Column(scale=1):
                save_button = gr.Button("Save", variant="primary")

        save_status = gr.Markdown()
        saved_file = gr.File(label="Saved Checkpoint", interactive=False)

        load_button.click(
            fn=load_for_ui,
            inputs=[path_input, upload_input],
            outputs=[
                checkpoint_state,
                source_path_state,
                load_info,
                overview_table,
                rename_table,
                preview_key,
                state_dict_table,
                content_table,
                output_path,
            ],
        )
        preview_key.change(
            fn=update_preview,
            inputs=[checkpoint_state, preview_key],
            outputs=[state_dict_table],
        )
        content_filter_button.click(
            fn=update_content_rows,
            inputs=[checkpoint_state, content_filter, content_max_rows],
            outputs=[content_table],
        )
        save_button.click(
            fn=save_checkpoint,
            inputs=[checkpoint_state, rename_table, output_path, source_path_state],
            outputs=[save_status, saved_file],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Gradio GUI for checkpoint inspection and key editing.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the Gradio app to.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind the Gradio app to.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share mode.")
    args = parser.parse_args()

    if gr is None:
        raise SystemExit("gradio is not installed. Install it with: python3 -m pip install gradio")

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
