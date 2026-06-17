from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"


def _import_model_loader() -> types.ModuleType:
    sys.path.insert(0, str(PACKAGE_SRC))
    try:
        return importlib.import_module("ltx_audio_trainer.model_loader")
    finally:
        sys.path.pop(0)


def _record_call(called: list[str], name: str, result: object) -> Callable[..., object]:
    def _loader(*_args: object, **_kwargs: object) -> object:
        called.append(name)
        return result

    return _loader


def _fail_loader(message: str) -> Callable[..., object]:
    def _loader(*_args: object, **_kwargs: object) -> object:
        pytest.fail(message)

    return _loader


def test_load_text_encoder_delegates_to_8bit_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_loader = _import_model_loader()
    gemma_root = tmp_path / "gemma"
    gemma_root.mkdir()

    captured: dict[str, object] = {}
    stub_module = types.ModuleType("ltx_audio_trainer.gemma_8bit")

    def _load_8bit_gemma(
        gemma_model_path: Path,
        dtype: object,
        device: object = None,
    ) -> str:
        captured["gemma_model_path"] = gemma_model_path
        captured["dtype"] = dtype
        captured["device"] = device
        return "8bit-text-encoder"

    stub_module.load_8bit_gemma = _load_8bit_gemma
    monkeypatch.setitem(sys.modules, "ltx_audio_trainer.gemma_8bit", stub_module)

    result = model_loader.load_text_encoder(gemma_root, device="cpu", load_in_8bit=True)

    assert result == "8bit-text-encoder"
    assert captured["gemma_model_path"] == gemma_root


def test_text_conditioning_components_unload_text_encoder() -> None:
    model_loader = _import_model_loader()

    text_encoder = types.SimpleNamespace(model=object(), tokenizer=object(), processor=object())
    components = model_loader.TextConditioningComponents(
        text_encoder=text_encoder,
        embeddings_processor="processor",
    )

    components.unload_text_encoder()

    assert components.text_encoder is text_encoder
    assert components.text_encoder.model is None
    assert components.text_encoder.tokenizer is None
    assert components.text_encoder.processor is None
    assert components.embeddings_processor == "processor"


def test_load_training_components_only_loads_training_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    model_loader = _import_model_loader()
    checkpoint_path = Path(__file__).resolve()

    called: list[str] = []

    monkeypatch.setattr(model_loader, "load_transformer", _record_call(called, "transformer", "t"))
    monkeypatch.setattr(
        model_loader,
        "load_text_conditioning_components",
        _record_call(called, "text", "text-stack"),
    )
    monkeypatch.setattr(
        model_loader,
        "load_audio_vae_encoder",
        _fail_loader("training components should not load audio encoder"),
    )
    monkeypatch.setattr(
        model_loader,
        "load_audio_vae_decoder",
        _fail_loader("training components should not load audio decoder"),
    )
    monkeypatch.setattr(
        model_loader,
        "load_vocoder",
        _fail_loader("training components should not load vocoder"),
    )

    scheduler_module = types.ModuleType("ltx_audio_core.components.schedulers")
    scheduler_module.LTX2Scheduler = lambda: "scheduler"
    monkeypatch.setitem(sys.modules, "ltx_audio_core.components.schedulers", scheduler_module)

    components = model_loader.load_training_components(
        checkpoint_path=checkpoint_path,
        text_encoder_path=checkpoint_path.parent,
    )

    assert components.transformer == "t"
    assert components.scheduler == "scheduler"
    assert components.text_conditioning == "text-stack"
    assert called == ["transformer", "text"]


def test_load_preprocess_components_loads_audio_encoder_and_text_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    model_loader = _import_model_loader()
    checkpoint_path = Path(__file__).resolve()

    called: list[str] = []

    monkeypatch.setattr(
        model_loader,
        "load_audio_vae_encoder",
        _record_call(called, "audio-encoder", "audio-encoder"),
    )
    monkeypatch.setattr(
        model_loader,
        "load_text_conditioning_components",
        _record_call(called, "text", "text-stack"),
    )
    monkeypatch.setattr(
        model_loader,
        "load_transformer",
        _fail_loader("preprocess components should not load transformer"),
    )

    components = model_loader.load_preprocess_components(
        checkpoint_path=checkpoint_path,
        text_encoder_path=checkpoint_path.parent,
    )

    assert components.audio_vae_encoder == "audio-encoder"
    assert components.text_conditioning == "text-stack"
    assert called == ["audio-encoder", "text"]


def test_load_validation_components_loads_decoder_vocoder_and_text_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_loader = _import_model_loader()
    checkpoint_path = Path(__file__).resolve()

    called: list[str] = []

    monkeypatch.setattr(model_loader, "load_transformer", _record_call(called, "transformer", "t"))
    monkeypatch.setattr(
        model_loader,
        "load_audio_vae_decoder",
        _record_call(called, "audio-decoder", "audio-decoder"),
    )
    monkeypatch.setattr(
        model_loader,
        "load_vocoder",
        _record_call(called, "vocoder", "vocoder"),
    )
    monkeypatch.setattr(
        model_loader,
        "load_text_conditioning_components",
        _record_call(called, "text", "text-stack"),
    )

    scheduler_module = types.ModuleType("ltx_audio_core.components.schedulers")
    scheduler_module.LTX2Scheduler = lambda: "scheduler"
    monkeypatch.setitem(sys.modules, "ltx_audio_core.components.schedulers", scheduler_module)

    components = model_loader.load_validation_components(
        checkpoint_path=checkpoint_path,
        text_encoder_path=checkpoint_path.parent,
    )

    assert components.transformer == "t"
    assert components.audio_vae_decoder == "audio-decoder"
    assert components.vocoder == "vocoder"
    assert components.scheduler == "scheduler"
    assert components.text_conditioning == "text-stack"
    assert called == ["transformer", "audio-decoder", "vocoder", "text"]
