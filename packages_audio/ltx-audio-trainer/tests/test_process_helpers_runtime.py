from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CAPTIONS_SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "process_captions.py"
AUDIO_SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "process_audio.py"


def _load_module(module_name: str, module_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_typer_stub() -> types.SimpleNamespace:
    def _identity(*_args: object, **_kwargs: object) -> None:
        return None

    def _command(*_args: object, **_kwargs: object) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            return function

        return _decorator

    def _typer(*_args: object, **_kwargs: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(command=_command)

    return types.SimpleNamespace(Typer=_typer, Argument=_identity, Option=_identity)


def _make_gpu_context_stub() -> type:
    class _Context:
        def __enter__(self) -> "_Context":
            return self

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc: BaseException | None,
            _tb: TracebackType | None,
        ) -> bool:
            return False

    return _Context


def _load_audio_helpers_module(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    logger: types.SimpleNamespace,
) -> types.ModuleType:
    monkeypatch.setitem(
        sys.modules,
        "ltx_audio_trainer",
        types.SimpleNamespace(logger=logger),
    )
    monkeypatch.setitem(
        sys.modules,
        "ltx_audio_core.model.audio_vae.audio_vae",
        types.SimpleNamespace(
            decode_audio=lambda *_args, **_kwargs: None,
            encode_audio=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ltx_audio_core.types",
        types.SimpleNamespace(Audio=type("Audio", (), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "ltx_audio_trainer.model_loader",
        types.SimpleNamespace(
            load_audio_vae_decoder=lambda *_args, **_kwargs: None,
            load_preprocess_components=lambda **_kwargs: None,
            load_vocoder=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "torchaudio", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "typer", _make_typer_stub())
    return _load_module(module_name, AUDIO_SCRIPT_PATH)


def test_atomic_save_replaces_target_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer_module = types.ModuleType("ltx_audio_trainer")
    trainer_module.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_a, **_k: None)
    monkeypatch.setitem(sys.modules, "ltx_audio_trainer", trainer_module)
    monkeypatch.setitem(
        sys.modules,
        "ltx_audio_trainer.model_loader",
        types.SimpleNamespace(load_text_conditioning_components=lambda **_kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "typer", _make_typer_stub())
    captions_module = _load_module("ltx_audio_trainer_process_captions_helpers", CAPTIONS_SCRIPT_PATH)

    target = tmp_path / "sample.pt"
    captions_module._atomic_save({"value": torch.tensor([1])}, target)

    loaded = torch.load(target, map_location="cpu", weights_only=True)
    assert loaded["value"].item() == 1
    assert not list(tmp_path.glob("*.tmp.*"))


def test_build_sharded_dataloader_filters_done_items(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer_module = types.ModuleType("ltx_audio_trainer")
    trainer_module.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_a, **_k: None)
    monkeypatch.setitem(sys.modules, "ltx_audio_trainer", trainer_module)
    monkeypatch.setitem(
        sys.modules,
        "ltx_audio_trainer.model_loader",
        types.SimpleNamespace(load_text_conditioning_components=lambda **_kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "typer", _make_typer_stub())

    accelerate_module = types.ModuleType("accelerate")
    accelerate_module.PartialState = lambda: types.SimpleNamespace(process_index=0, num_processes=2)
    monkeypatch.setitem(sys.modules, "accelerate", accelerate_module)

    captions_module = _load_module("ltx_audio_trainer_process_captions_shard", CAPTIONS_SCRIPT_PATH)

    dataset = [{"index": idx} for idx in range(6)]
    dataloader = captions_module._build_sharded_dataloader(
        dataset,
        batch_size=2,
        num_workers=0,
        is_done=lambda idx: idx == 2,
        overwrite=False,
    )

    assert dataloader is not None
    indices = []
    for batch in dataloader:
        indices.extend(batch["index"].tolist())
    assert indices == [0, 4]


def test_process_audio_dataset_forwards_decode_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([{"caption": "rain", "audio_path": "a.wav"}]), encoding="utf-8")

    calls: dict[str, dict[str, object]] = {}

    monkeypatch.setitem(
        sys.modules,
        "process_captions",
        types.SimpleNamespace(compute_captions_embeddings=lambda **kwargs: calls.setdefault("captions", kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "process_audio",
        types.SimpleNamespace(compute_audio_latents=lambda **kwargs: calls.setdefault("audio", kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "ltx_audio_trainer.gpu_utils",
        types.SimpleNamespace(free_gpu_memory_context=_make_gpu_context_stub()),
    )
    trainer_module = types.ModuleType("ltx_audio_trainer")
    trainer_module.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "ltx_audio_trainer", trainer_module)
    monkeypatch.setitem(sys.modules, "rich.console", types.SimpleNamespace(Console=type("Console", (), {})))
    monkeypatch.setitem(sys.modules, "typer", _make_typer_stub())

    module = _load_module(
        "ltx_audio_trainer_process_audio_dataset_decode",
        PACKAGE_ROOT / "scripts" / "process_audio_dataset.py",
    )
    module.preprocess_dataset(
        dataset_file=str(dataset_path),
        caption_column="caption",
        audio_column="audio_path",
        batch_size=1,
        output_dir=None,
        lora_trigger=None,
        model_path=str(dataset_path),
        text_encoder_path=str(tmp_path),
        device="cpu",
        remove_llm_prefixes=False,
        load_text_encoder_in_8bit=False,
        overwrite=False,
        decode=True,
    )

    assert calls["audio"]["decode"] is True


def test_append_failure_record_appends_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer_module = types.ModuleType("ltx_audio_trainer")
    trainer_module.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_a, **_k: None)
    monkeypatch.setitem(sys.modules, "ltx_audio_trainer", trainer_module)
    monkeypatch.setitem(
        sys.modules,
        "ltx_audio_trainer.model_loader",
        types.SimpleNamespace(load_text_conditioning_components=lambda **_kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "typer", _make_typer_stub())
    captions_module = _load_module("ltx_audio_trainer_process_captions_failures", CAPTIONS_SCRIPT_PATH)

    manifest_path = tmp_path / "caption_failures.jsonl"
    captions_module._append_failure_record(
        manifest_path,
        {
            "input_path": "a.wav",
            "output_path": "a.pt",
            "error": "boom",
        },
    )
    captions_module._append_failure_record(
        manifest_path,
        {
            "input_path": "b.wav",
            "output_path": "b.pt",
            "error": "bad prompt",
        },
    )

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["input_path"] == "a.wav"
    assert json.loads(lines[1])["error"] == "bad prompt"


def test_save_decode_failure_artifact_writes_debug_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_a, **_k: None)
    audio_module = _load_audio_helpers_module("ltx_audio_trainer_process_audio_debug", monkeypatch, logger)

    artifact_path = audio_module._save_decode_failure_artifact(
        latent=torch.ones(2, 3, 4),
        output_dir=tmp_path,
        relative_path=Path("clips/sample.wav"),
        error_message="decoded audio contains nan",
    )

    saved = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert artifact_path == tmp_path / "_decode_failures" / "clips" / "sample.pt"
    assert saved["error"] == "decoded audio contains nan"
    assert tuple(saved["latents"].shape) == (2, 3, 4)


def test_log_processing_summary_reports_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    info_messages: list[str] = []
    warning_messages: list[str] = []
    logger = types.SimpleNamespace(
        info=lambda message: info_messages.append(message),
        warning=lambda message: warning_messages.append(message),
    )
    audio_module = _load_audio_helpers_module("ltx_audio_trainer_process_audio_summary", monkeypatch, logger)

    audio_module._log_processing_summary(
        modality="audio latents",
        processed=3,
        skipped=2,
        failed=1,
        output_dir=Path("/tmp/out"),
        failure_manifest=Path("/tmp/out/audio_failures.jsonl"),
    )

    assert any("processed=3" in message for message in info_messages)
    assert any("skipped=2" in message for message in info_messages)
    assert any("failed=1" in message for message in warning_messages)
