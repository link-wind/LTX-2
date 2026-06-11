from __future__ import annotations

import ast
from pathlib import Path


def test_constants_module_exists_with_pipeline_params_and_detect_params() -> None:
    constants_path = (
        Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "utils" / "constants.py"
    )

    assert constants_path.exists(), f"Missing constants module: {constants_path}"

    module = ast.parse(constants_path.read_text(encoding="utf-8"))
    class_names = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
    function_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}

    assert "PipelineParams" in class_names
    assert "detect_params" in function_names
