from __future__ import annotations

import sys
import types
from collections.abc import Generator
from pathlib import Path

import pytest
import torch

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
CORE_SRC = Path(__file__).resolve().parents[2] / "ltx-audio-core" / "src"


@pytest.fixture
def validation_sampler_module() -> Generator[types.ModuleType, None, None]:
    sys.path.insert(0, str(PACKAGE_SRC))
    sys.path.insert(0, str(CORE_SRC))
    try:
        from ltx_audio_trainer import validation_sampler  # noqa: PLC0415

        yield validation_sampler
    finally:
        sys.path.pop(0)
        sys.path.pop(0)


def test_validation_sampler_generates_audio_from_cached_embeddings(
    validation_sampler_module: types.ModuleType,
) -> None:
    class FakeVelocityModel(torch.nn.Module):
        def forward(self, audio, perturbations=None) -> torch.Tensor:  # noqa: ANN001, ARG002
            return torch.zeros_like(audio.latent)

    class FakeAudioDecoder(torch.nn.Module):
        def forward(self, latent) -> torch.Tensor:  # noqa: ANN001
            return latent

    class FakeVocoder(torch.nn.Module):
        output_sampling_rate = 24_000

        def forward(self, decoded) -> torch.Tensor:  # noqa: ANN001
            return decoded.mean(dim=1).reshape(decoded.shape[0], 1, -1)

    cached = validation_sampler_module.CachedPromptEmbeddings(
        audio_context_positive=torch.ones(1, 4, 2048),
        audio_context_negative=torch.zeros(1, 4, 2048),
        audio_attention_mask_positive=torch.ones(1, 4, dtype=torch.bool),
        audio_attention_mask_negative=torch.ones(1, 4, dtype=torch.bool),
    )
    config = validation_sampler_module.GenerationConfig(
        prompt="rain on windows",
        negative_prompt="",
        audio_duration_seconds=0.25,
        num_inference_steps=2,
        guidance_scale=3.0,
        seed=0,
        cached_embeddings=cached,
    )

    sampler = validation_sampler_module.ValidationSampler(
        transformer=FakeVelocityModel(),
        audio_decoder=FakeAudioDecoder(),
        vocoder=FakeVocoder(),
        scheduler=validation_sampler_module.LTX2Scheduler(),
    )
    audio = sampler.generate(config, device="cpu")

    assert audio.sampling_rate == 24_000
    assert audio.waveform.ndim == 2
    assert audio.waveform.numel() > 0


def test_validation_sampler_generates_audio_from_prompt_encoding(
    validation_sampler_module: types.ModuleType,
) -> None:
    class FakeVelocityModel(torch.nn.Module):
        def forward(self, audio, perturbations=None) -> torch.Tensor:  # noqa: ANN001, ARG002
            return torch.zeros_like(audio.latent)

    class FakeAudioDecoder(torch.nn.Module):
        def forward(self, latent) -> torch.Tensor:  # noqa: ANN001
            return latent

    class FakeVocoder(torch.nn.Module):
        output_sampling_rate = 24_000

        def forward(self, decoded) -> torch.Tensor:  # noqa: ANN001
            return decoded.mean(dim=1).reshape(decoded.shape[0], 1, -1)

    class FakeTextEncoder:
        def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
            value = 1.0 if prompt else 0.0
            return torch.full((1, 4, 8), value), torch.ones(1, 4, dtype=torch.bool)

    class FakeEmbeddingsProcessor:
        def process_hidden_states(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                audio_encoding=hidden_states.repeat(1, 1, 256),
                video_encoding=None,
                attention_mask=attention_mask,
            )

    config = validation_sampler_module.GenerationConfig(
        prompt="thunder rolling far away",
        negative_prompt="",
        audio_duration_seconds=0.25,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=0,
        cached_embeddings=None,
    )

    sampler = validation_sampler_module.ValidationSampler(
        transformer=FakeVelocityModel(),
        audio_decoder=FakeAudioDecoder(),
        vocoder=FakeVocoder(),
        text_encoder=FakeTextEncoder(),
        embeddings_processor=FakeEmbeddingsProcessor(),
        scheduler=validation_sampler_module.LTX2Scheduler(),
    )
    audio = sampler.generate(config, device="cpu")

    assert audio.sampling_rate == 24_000
    assert audio.waveform.ndim == 2
    assert audio.waveform.numel() > 0
