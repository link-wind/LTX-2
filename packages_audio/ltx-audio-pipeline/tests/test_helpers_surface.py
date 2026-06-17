from __future__ import annotations

import ast
from pathlib import Path


def test_helpers_module_exists_with_required_functions() -> None:
    helpers_path = (
        Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "utils" / "helpers.py"
    )

    assert helpers_path.exists(), f"Missing helpers module: {helpers_path}"

    module = ast.parse(helpers_path.read_text(encoding="utf-8"))
    function_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}

    assert {
        "cleanup_memory",
        "clean_response",
        "generate_enhanced_prompt",
        "modality_from_latent_state",
        "post_process_latent",
        "timesteps_from_mask",
    }.issubset(function_names)
