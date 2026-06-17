from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = PACKAGE_ROOT / "src"
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "inference.py"


def _load_inference_module(module_name: str, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.syspath_prepend(str(PACKAGE_SRC))
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Failed to create inference.py module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inference_main_generates_audio_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torchaudio_module = types.ModuleType("torchaudio")
    saved_calls: list[tuple[str, tuple[int, ...], int]] = []

    def _save(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
        saved_calls.append((path, tuple(waveform.shape), sample_rate))

    torchaudio_module.save = _save
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_module)

    inference_module = _load_inference_module("ltx_audio_trainer_inference_runtime", monkeypatch)

    class _Progress:
        def __init__(self, num_steps: int, description: str = "Generating") -> None:
            self.num_steps = num_steps
            self.description = description

        def __enter__(self) -> "_Progress":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def advance_step(self) -> None:
            return None

    class _Sampler:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def generate(self, config, device: str) -> types.SimpleNamespace:  # noqa: ANN001
            assert config.prompt == "rain in the city"
            assert config.audio_duration_seconds == pytest.approx(1.5)
            assert config.num_inference_steps == 4
            assert config.guidance_scale == pytest.approx(3.5)
            assert config.cached_embeddings is None
            assert device == "cpu"
            return types.SimpleNamespace(
                waveform=torch.ones(1, 64),
                sampling_rate=24_000,
            )

    components = types.SimpleNamespace(
        transformer="transformer",
        audio_vae_decoder="audio-decoder",
        vocoder="vocoder",
        text_encoder="text-encoder",
        embeddings_processor="embeddings-processor",
    )

    monkeypatch.setattr(inference_module, "StandaloneSamplingProgress", _Progress)
    monkeypatch.setattr(inference_module, "ValidationSampler", _Sampler)
    monkeypatch.setattr(inference_module, "load_model", lambda **_kwargs: components)

    output_path = tmp_path / "sample.wav"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--checkpoint",
            str(tmp_path / "model.safetensors"),
            "--text-encoder-path",
            str(tmp_path / "gemma"),
            "--prompt",
            "rain in the city",
            "--audio-duration-seconds",
            "1.5",
            "--num-inference-steps",
            "4",
            "--guidance-scale",
            "3.5",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ],
    )

    inference_module.main()

    assert saved_calls == [(str(output_path), (1, 64), 24_000)]


def test_extract_lora_target_modules_returns_sorted_unique_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    torchaudio_module = types.ModuleType("torchaudio")
    torchaudio_module.save = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_module)
    inference_module = _load_inference_module("ltx_audio_trainer_inference_lora", monkeypatch)

    state_dict = {
        "diffusion_model.transformer_blocks.0.attn.to_k.lora_A.weight": torch.ones(2, 2),
        "diffusion_model.transformer_blocks.0.attn.to_k.lora_B.weight": torch.ones(2, 2),
        "diffusion_model.transformer_blocks.1.ff.proj.lora_A.weight": torch.ones(2, 2),
    }

    modules = inference_module.extract_lora_target_modules(state_dict)

    assert modules == [
        "diffusion_model.transformer_blocks.0.attn.to_k",
        "diffusion_model.transformer_blocks.1.ff.proj",
    ]
