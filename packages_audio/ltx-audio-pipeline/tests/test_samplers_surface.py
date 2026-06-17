from __future__ import annotations

import ast
from pathlib import Path


def test_samplers_module_has_audio_only_compatibility_docstring() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ltx_audio_pipeline"
        / "utils"
        / "samplers.py"
    )
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    docstring = ast.get_docstring(module)

    assert isinstance(docstring, str)
    assert "audio-only" in docstring
