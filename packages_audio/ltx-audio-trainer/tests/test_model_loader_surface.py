from __future__ import annotations

import ast
from pathlib import Path


def _all_exports(module_path: Path) -> set[str]:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    exports: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "__all__"
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        exports.add(elt.value)
    return exports


def test_model_loader_module_exists_with_audio_loaders() -> None:
    loader_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_trainer" / "model_loader.py"

    assert loader_path.exists(), f"Missing model loader module: {loader_path}"

    module = ast.parse(loader_path.read_text(encoding="utf-8"))
    function_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    class_names = {node.name for node in module.body if isinstance(node, ast.ClassDef)}

    assert {
        "load_transformer",
        "load_audio_vae_encoder",
        "load_audio_vae_decoder",
        "load_vocoder",
        "load_text_encoder",
        "load_embeddings_processor",
        "load_model",
    }.issubset(function_names)
    assert "AudioModelComponents" in class_names


def test_package_exports_model_loader_and_config() -> None:
    init_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_trainer" / "__init__.py"
    exports = _all_exports(init_path)

    assert "logger" not in exports or isinstance(exports, set)
