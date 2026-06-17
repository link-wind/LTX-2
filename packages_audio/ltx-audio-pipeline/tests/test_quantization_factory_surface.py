from __future__ import annotations

import ast
from pathlib import Path


def test_quantization_factory_is_parseable_in_local_static_checks() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ltx_audio_pipeline"
        / "utils"
        / "quantization_factory.py"
    )

    ast.parse(module_path.read_text(encoding="utf-8"))
