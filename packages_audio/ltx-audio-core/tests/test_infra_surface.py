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


def test_block_streaming_package_exists_with_core_files() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_core" / "block_streaming"

    expected_files = {
        "__init__.py",
        "builder.py",
        "disk.py",
        "pool.py",
        "provider.py",
        "source.py",
        "utils.py",
        "wrapper.py",
    }
    actual_files = {path.name for path in root.glob("*.py")}

    assert expected_files.issubset(actual_files)


def test_transformer_init_exports_compiling_helpers() -> None:
    init_path = (
        Path(__file__).resolve().parents[1] / "src" / "ltx_audio_core" / "model" / "transformer" / "__init__.py"
    )
    exports = _all_exports(init_path)

    assert {"CompilationConfig", "build_compile_transformer_op", "modify_sd_ops_for_compilation"}.issubset(exports)
