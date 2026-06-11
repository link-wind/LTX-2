from __future__ import annotations

import ast
from pathlib import Path


def test_types_module_declares_transformer_like_protocol() -> None:
    types_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "utils" / "types.py"
    module = ast.parse(types_path.read_text(encoding="utf-8"))

    class_names = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
    assert "TransformerLike" in class_names


def test_types_exports_transformer_like() -> None:
    types_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_pipeline" / "utils" / "types.py"
    module = ast.parse(types_path.read_text(encoding="utf-8"))

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

    assert "TransformerLike" in exports
