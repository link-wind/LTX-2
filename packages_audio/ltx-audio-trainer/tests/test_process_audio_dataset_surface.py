from __future__ import annotations

import ast
from pathlib import Path


def test_process_audio_dataset_script_exists_with_main_functions() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "process_audio_dataset.py"

    assert script_path.exists(), f"Missing process_audio_dataset script: {script_path}"

    module = ast.parse(script_path.read_text(encoding="utf-8"))
    function_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert {"preprocess_dataset", "_validate_dataset_file", "main"}.issubset(function_names)


def test_process_audio_dataset_mentions_audio_and_caption_processing() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "process_audio_dataset.py"
    source = script_path.read_text(encoding="utf-8")

    assert "compute_captions_embeddings" in source
    assert "compute_audio_latents" in source
    assert "audio_column" in source
    assert ".precomputed" in source
