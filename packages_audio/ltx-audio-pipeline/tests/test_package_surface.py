from __future__ import annotations

import ast
from pathlib import Path


def test_top_level_package_exports_audio_entrypoints() -> None:
    init_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "__init__.py"
    module = ast.parse(init_path.read_text(encoding="utf-8"))

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

    assert {"AudioConditioner", "AudioDecoder", "DiffusionStage"}.issubset(exports)
