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
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exports.add(elt.value)
    return exports


def test_gemma_text_encoder_package_exists_with_core_files() -> None:
    gemma_root = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_core" / "text_encoders" / "gemma"

    expected_files = {
        "__init__.py",
        "config.py",
        "embeddings_connector.py",
        "embeddings_processor.py",
        "feature_extractor.py",
        "tokenizer.py",
        "encoders/base_encoder.py",
        "encoders/encoder_configurator.py",
    }

    actual_files = {
        str(path.relative_to(gemma_root)).replace("\\", "/")
        for path in gemma_root.rglob("*")
        if path.is_file()
    }
    assert expected_files.issubset(actual_files)


def test_gemma_package_exports_prompt_encoder_dependencies() -> None:
    gemma_init = (
        Path(__file__).resolve().parents[1] / "src" / "ltx_audio_core" / "text_encoders" / "gemma" / "__init__.py"
    )

    exports = _all_exports(gemma_init)

    assert {
        "EmbeddingsProcessor",
        "EmbeddingsProcessorConfigurator",
        "EmbeddingsProcessorOutput",
        "GemmaTextEncoder",
        "GemmaTextEncoderConfigurator",
        "module_ops_from_gemma_root",
    }.issubset(exports)
