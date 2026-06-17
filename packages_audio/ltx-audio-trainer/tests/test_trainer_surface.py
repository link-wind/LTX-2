from __future__ import annotations

import ast
from pathlib import Path


def _module_items(module_path: Path) -> tuple[set[str], set[str]]:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    classes = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
    functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    return classes, functions


def test_trainer_module_exposes_audio_trainer_surface() -> None:
    module_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_trainer" / "trainer.py"
    classes, _ = _module_items(module_path)

    assert {"AudioTrainer", "TrainingStats", "TrainingStepOutput"}.issubset(classes)


def test_train_script_exists_and_has_main() -> None:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    _, functions = _module_items(module_path)

    assert "main" in functions
