from __future__ import annotations

import ast
from pathlib import Path


def _module_ast(module_path: Path) -> ast.Module:
    return ast.parse(module_path.read_text(encoding="utf-8"))


def _class_node(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Missing class {class_name}")


def _all_exports(module_path: Path) -> set[str]:
    module = _module_ast(module_path)
    exports: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exports.add(elt.value)
    return exports


def _assigned_self_attrs(function: ast.FunctionDef) -> set[str]:
    attrs: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                attrs.add(target.attr)
    return attrs


def test_audio_one_stage_module_exists_with_pipeline_class() -> None:
    module_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "audio_one_stage.py"

    assert module_path.exists(), f"Missing audio pipeline module: {module_path}"

    module = _module_ast(module_path)
    _class_node(module, "AudioOneStagePipeline")
    assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in module.body)


def test_audio_one_stage_pipeline_keeps_core_pipeline_members() -> None:
    module_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "audio_one_stage.py"
    module = _module_ast(module_path)
    pipeline_class = _class_node(module, "AudioOneStagePipeline")

    init_node = next(
        node for node in pipeline_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assigned_attrs = _assigned_self_attrs(init_node)

    assert {"_scheduler", "stage", "audio_decoder"}.issubset(assigned_attrs)


def test_top_level_package_exports_audio_one_stage_pipeline() -> None:
    init_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "__init__.py"

    exports = _all_exports(init_path)

    assert "AudioOneStagePipeline" in exports
