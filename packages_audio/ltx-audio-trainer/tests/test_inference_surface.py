from __future__ import annotations

import ast
from pathlib import Path


def _module_items(module_path: Path) -> tuple[set[str], set[str]]:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    classes = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
    functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    return classes, functions


def test_inference_script_exists_and_has_expected_functions() -> None:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "inference.py"
    _, functions = _module_items(module_path)

    assert {"main", "extract_lora_target_modules", "load_lora_weights"}.issubset(functions)
