from __future__ import annotations

import ast
from pathlib import Path


def test_config_display_module_exists_with_print_config() -> None:
    module_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_trainer" / "config_display.py"

    assert module_path.exists(), f"Missing config display module: {module_path}"

    module = ast.parse(module_path.read_text(encoding="utf-8"))
    function_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert "print_config" in function_names


def test_config_display_mentions_audio_validation_fields() -> None:
    module_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_trainer" / "config_display.py"
    source = module_path.read_text(encoding="utf-8")

    assert "audio_duration_seconds" in source
    assert "load_text_encoder_in_8bit" in source
    assert "preprocessed_data_root" in source


def test_hf_hub_utils_module_exists_with_audio_push_helpers() -> None:
    module_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_trainer" / "hf_hub_utils.py"

    assert module_path.exists(), f"Missing hf hub utils module: {module_path}"

    module = ast.parse(module_path.read_text(encoding="utf-8"))
    function_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert {"push_to_hub", "_create_model_card", "_copy_audio_samples"}.issubset(function_names)


def test_hf_hub_utils_mentions_audio_samples_and_wav() -> None:
    module_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_trainer" / "hf_hub_utils.py"
    source = module_path.read_text(encoding="utf-8")

    assert "sample_" in source
    assert ".wav" in source
    assert "validation audio" in source.lower() or "audio samples" in source.lower()
