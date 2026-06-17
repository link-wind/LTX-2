from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "process_audio_dataset.py"


def _load_module(module_name: str, module_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _noop(*_args: object, **_kwargs: object) -> None:
    return None


def _none_option(*_args: object, **_kwargs: object) -> None:
    return None


def test_preprocess_dataset_routes_outputs_to_precomputed_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "caption": "soft piano and rain",
                    "audio_path": "samples/audio.wav",
                }
            ]
        ),
        encoding="utf-8",
    )

    calls: dict[str, dict[str, object]] = {}

    captions_module = types.ModuleType("process_captions")

    def _compute_captions_embeddings(**kwargs: object) -> None:
        calls["captions"] = kwargs

    captions_module.compute_captions_embeddings = _compute_captions_embeddings
    monkeypatch.setitem(sys.modules, "process_captions", captions_module)

    audio_module = types.ModuleType("process_audio")

    def _compute_audio_latents(**kwargs: object) -> None:
        calls["audio"] = kwargs

    audio_module.compute_audio_latents = _compute_audio_latents
    monkeypatch.setitem(sys.modules, "process_audio", audio_module)

    gpu_utils_module = types.ModuleType("ltx_audio_trainer.gpu_utils")

    class _FreeMemoryContext:
        def __enter__(self) -> "_FreeMemoryContext":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    gpu_utils_module.free_gpu_memory_context = _FreeMemoryContext
    monkeypatch.setitem(sys.modules, "ltx_audio_trainer.gpu_utils", gpu_utils_module)

    trainer_module = types.ModuleType("ltx_audio_trainer")
    trainer_module.logger = types.SimpleNamespace(info=_noop)
    monkeypatch.setitem(sys.modules, "ltx_audio_trainer", trainer_module)

    rich_console_module = types.ModuleType("rich.console")
    rich_console_module.Console = type("Console", (), {})
    monkeypatch.setitem(sys.modules, "rich.console", rich_console_module)

    typer_module = types.ModuleType("typer")

    class _Typer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def command(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> Callable[[Callable[..., object]], Callable[..., object]]:
            def _decorator(fn: Callable[..., object]) -> Callable[..., object]:
                return fn

            return _decorator

    typer_module.Typer = _Typer
    typer_module.Argument = _none_option
    typer_module.Option = _none_option
    monkeypatch.setitem(sys.modules, "typer", typer_module)

    module = _load_module("ltx_audio_trainer_process_audio_dataset_test", SCRIPT_PATH)

    module.preprocess_dataset(
        dataset_file=str(dataset_path),
        caption_column="caption",
        audio_column="audio_path",
        batch_size=2,
        output_dir=None,
        lora_trigger=None,
        model_path=str(dataset_path),
        text_encoder_path=str(tmp_path),
        device="cpu",
        remove_llm_prefixes=False,
        load_text_encoder_in_8bit=False,
        overwrite=False,
    )

    assert calls["captions"]["output_dir"].endswith(".precomputed/conditions")
    assert calls["audio"]["output_dir"].endswith(".precomputed/audio_latents")
    assert calls["captions"]["media_column"] == "audio_path"
    assert calls["audio"]["audio_column"] == "audio_path"
