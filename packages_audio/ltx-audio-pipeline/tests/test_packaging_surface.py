from __future__ import annotations

from pathlib import Path


def test_package_readme_exists_and_mentions_audio_only() -> None:
    readme_path = Path(__file__).resolve().parents[1] / "README.md"

    assert readme_path.exists(), f"Missing package README: {readme_path}"

    readme_text = readme_path.read_text(encoding="utf-8").lower()
    assert "audio-only" in readme_text
    assert "pipeline" in readme_text
    assert "audioonestagepipeline" in readme_text
    assert "--offload" in readme_text
    assert "--compile" in readme_text
