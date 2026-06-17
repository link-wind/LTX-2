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


def test_utils_exports_configuration_enums() -> None:
    exports = _all_exports(
        Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "utils" / "__init__.py"
    )
    assert {"OffloadMode", "QuantizationKind"}.issubset(exports)


def test_top_level_package_exports_configuration_enums() -> None:
    exports = _all_exports(
        Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "__init__.py"
    )
    assert {"OffloadMode", "QuantizationKind"}.issubset(exports)
