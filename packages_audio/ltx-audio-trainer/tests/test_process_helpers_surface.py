from __future__ import annotations

import ast
from pathlib import Path


def _module_functions(module_path: Path) -> set[str]:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    return {node.name for node in module.body if isinstance(node, ast.FunctionDef)}


def test_process_captions_exposes_atomic_and_sharded_helpers() -> None:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "process_captions.py"
    function_names = _module_functions(module_path)

    assert {
        "_atomic_save",
        "_build_sharded_dataloader",
        "_retry",
        "_append_failure_record",
        "_log_processing_summary",
    }.issubset(function_names)


def test_process_audio_exposes_atomic_sharded_and_decode_helpers() -> None:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "process_audio.py"
    function_names = _module_functions(module_path)

    assert {
        "_atomic_save",
        "_build_sharded_dataloader",
        "_retry",
        "_decode_and_validate",
        "_append_failure_record",
        "_save_decode_failure_artifact",
        "_log_processing_summary",
    }.issubset(function_names)
