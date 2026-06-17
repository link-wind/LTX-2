from __future__ import annotations

import ast
from pathlib import Path


def test_args_module_exists_with_audio_cli_builders() -> None:
    args_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "utils" / "args.py"

    assert args_path.exists(), f"Missing args module: {args_path}"

    module = ast.parse(args_path.read_text(encoding="utf-8"))
    function_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    class_names = {node.name for node in module.body if isinstance(node, ast.ClassDef)}

    assert {
        "resolve_path",
        "resolve_existing_path",
        "detect_checkpoint_path",
        "basic_arg_parser",
        "audio_one_stage_arg_parser",
    }.issubset(function_names)
    assert "CompileAction" in class_names
