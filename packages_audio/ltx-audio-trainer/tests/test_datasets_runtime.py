from __future__ import annotations

import sys
import types
from collections.abc import Generator
from pathlib import Path

import pytest
import torch

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def dataset_module() -> Generator[types.ModuleType, None, None]:
    sys.path.insert(0, str(PACKAGE_SRC))
    try:
        from ltx_audio_trainer import datasets  # noqa: PLC0415

        yield datasets
    finally:
        sys.path.pop(0)


def test_precomputed_dataset_loads_audio_latents_and_conditions(
    tmp_path: Path,
    dataset_module: types.ModuleType,
) -> None:
    precomputed_root = tmp_path / ".precomputed"
    audio_latents_dir = precomputed_root / "audio_latents"
    conditions_dir = precomputed_root / "conditions"
    audio_latents_dir.mkdir(parents=True)
    conditions_dir.mkdir(parents=True)

    torch.save(
        {
            "latents": torch.randn(8, 12, 16),
            "num_frames": 12,
            "mel_bins": 16,
        },
        audio_latents_dir / "sample_000.pt",
    )
    torch.save(
        {
            "video_prompt_embeds": torch.randn(32, 2048),
            "audio_prompt_embeds": torch.randn(32, 2048),
            "prompt_attention_mask": torch.ones(32, dtype=torch.bool),
        },
        conditions_dir / "sample_000.pt",
    )

    dataset = dataset_module.PrecomputedDataset(str(tmp_path))
    item = dataset[0]

    assert item["latent_conditions"]["latents"].shape == (8, 12, 16)
    assert item["text_conditions"]["audio_prompt_embeds"].shape == (32, 2048)
    assert item["idx"] == 0


def test_precomputed_dataset_supports_legacy_condition_file_names(
    tmp_path: Path,
    dataset_module: types.ModuleType,
) -> None:
    precomputed_root = tmp_path / ".precomputed"
    audio_latents_dir = precomputed_root / "audio_latents"
    conditions_dir = precomputed_root / "conditions"
    audio_latents_dir.mkdir(parents=True)
    conditions_dir.mkdir(parents=True)

    torch.save(
        {
            "latents": torch.randn(8, 10, 16),
            "num_frames": 10,
            "mel_bins": 16,
        },
        audio_latents_dir / "latent_001.pt",
    )
    torch.save(
        {
            "audio_prompt_embeds": torch.randn(16, 2048),
            "prompt_attention_mask": torch.ones(16, dtype=torch.bool),
        },
        conditions_dir / "condition_001.pt",
    )

    dataset = dataset_module.PrecomputedDataset(str(tmp_path))

    assert len(dataset) == 1
    assert dataset[0]["text_conditions"]["audio_prompt_embeds"].shape == (16, 2048)


def test_precomputed_dataset_unpatchifies_legacy_audio_tokens(
    tmp_path: Path,
    dataset_module: types.ModuleType,
) -> None:
    precomputed_root = tmp_path / ".precomputed"
    audio_latents_dir = precomputed_root / "audio_latents"
    conditions_dir = precomputed_root / "conditions"
    audio_latents_dir.mkdir(parents=True)
    conditions_dir.mkdir(parents=True)

    torch.save(
        {
            "latents": torch.randn(6, 128),
            "num_frames": 6,
            "latent_channels": 8,
            "mel_bins": 16,
        },
        audio_latents_dir / "sample_legacy.pt",
    )
    torch.save(
        {
            "audio_prompt_embeds": torch.randn(8, 2048),
            "prompt_attention_mask": torch.ones(8, dtype=torch.bool),
        },
        conditions_dir / "sample_legacy.pt",
    )

    dataset = dataset_module.PrecomputedDataset(str(tmp_path))
    item = dataset[0]

    assert item["latent_conditions"]["latents"].shape == (8, 6, 16)
