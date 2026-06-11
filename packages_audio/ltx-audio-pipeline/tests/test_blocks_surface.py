from __future__ import annotations

import ast
from pathlib import Path


def _class_names(module_path: Path) -> set[str]:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    return {node.name for node in module.body if isinstance(node, ast.ClassDef)}


def _all_exports(module_path: Path) -> set[str]:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        values: set[str] = set()
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                values.add(elt.value)
                        return values
    return set()


def test_blocks_module_exists_with_audio_classes() -> None:
    blocks_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "utils" / "blocks.py"

    assert blocks_path.exists(), f"Missing blocks module: {blocks_path}"
    assert {"DiffusionStage", "AudioDecoder", "AudioConditioner", "PromptEncoder"}.issubset(_class_names(blocks_path))


def test_utils_init_exports_audio_blocks() -> None:
    init_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "utils" / "__init__.py"

    exports = _all_exports(init_path)

    assert {"DiffusionStage", "AudioDecoder", "AudioConditioner", "PromptEncoder"}.issubset(exports)


def test_blocks_module_references_streaming_and_compilation_support() -> None:
    blocks_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "utils" / "blocks.py"
    source = blocks_path.read_text(encoding="utf-8")

    assert "StreamingModelBuilder" in source
    assert "CompilationConfig" in source
    assert "_streaming_text_encoder_builder" in source
    assert "offload_mode is not yet supported for PromptEncoder" not in source
