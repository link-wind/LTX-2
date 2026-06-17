from __future__ import annotations

from pathlib import Path


def test_denoisers_do_not_reference_missing_audio_core_batch_split_module() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ltx_audio_pipeline"
        / "utils"
        / "denoisers.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "ltx_audio_core.batch_split.BatchSplitAdapter" not in source
