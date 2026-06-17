# ruff: noqa: PLC0415

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ltx_audio_trainer import logger

Device = str | torch.device

if TYPE_CHECKING:
    from ltx_audio_core.components.schedulers import LTX2Scheduler
    from ltx_audio_core.model.audio_vae.audio_vae import AudioDecoder, AudioEncoder
    from ltx_audio_core.model.audio_vae.vocoder import Vocoder
    from ltx_audio_core.model.transformer.model import LTXModel
    from ltx_audio_core.text_encoders.gemma import EmbeddingsProcessor, GemmaTextEncoder


def _to_torch_device(device: Device) -> torch.device:
    return torch.device(device) if isinstance(device, str) else device


@dataclass
class TextConditioningComponents:
    text_encoder: "GemmaTextEncoder | None" = None
    embeddings_processor: "EmbeddingsProcessor | None" = None

    def require_text_encoder(self) -> "GemmaTextEncoder":
        if self.text_encoder is None:
            raise ValueError("Text encoder is not loaded")
        return self.text_encoder

    def require_embeddings_processor(self) -> "EmbeddingsProcessor":
        if self.embeddings_processor is None:
            raise ValueError("Embeddings processor is not loaded")
        return self.embeddings_processor

    def unload_text_encoder(self, free_memory: bool = False) -> None:
        if self.text_encoder is None:
            return

        self.text_encoder.model = None
        self.text_encoder.tokenizer = None
        self.text_encoder.processor = None

        if free_memory:
            from ltx_audio_trainer.gpu_utils import free_gpu_memory

            free_gpu_memory()


def load_transformer(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "LTXModel":
    from ltx_audio_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_audio_core.model.transformer import LTX_AUDIO_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTX_AUDIO_MODEL_COMFY_RENAMING_MAP,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioEncoder":
    from ltx_audio_core.loader import SingleGPUModelBuilder
    from ltx_audio_core.model.audio_vae.model_configurator import (
        AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        AudioEncoderConfigurator,
    )

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioDecoder":
    from ltx_audio_core.loader import SingleGPUModelBuilder
    from ltx_audio_core.model.audio_vae.model_configurator import (
        AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
        AudioDecoderConfigurator,
    )

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_vocoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "Vocoder":
    from ltx_audio_core.loader import SingleGPUModelBuilder
    from ltx_audio_core.model.audio_vae.model_configurator import VOCODER_COMFY_KEYS_FILTER, VocoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VocoderConfigurator,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_text_encoder(
    gemma_model_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
) -> "GemmaTextEncoder":
    if not Path(gemma_model_path).is_dir():
        raise ValueError(f"Gemma model path is not a directory: {gemma_model_path}")

    if load_in_8bit:
        from ltx_audio_trainer.gemma_8bit import load_8bit_gemma

        return load_8bit_gemma(gemma_model_path, dtype, device=device)

    from ltx_audio_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_audio_core.text_encoders.gemma import (
        GEMMA_LLM_KEY_OPS,
        GEMMA_MODEL_OPS,
        GemmaTextEncoderConfigurator,
        module_ops_from_gemma_root,
    )
    from ltx_audio_core.utils import find_matching_file

    torch_device = _to_torch_device(device)
    gemma_model_folder = find_matching_file(str(gemma_model_path), "model*.safetensors").parent
    gemma_weight_paths = [str(path) for path in gemma_model_folder.rglob("*.safetensors")]

    return SingleGPUModelBuilder(
        model_path=tuple(gemma_weight_paths),
        model_class_configurator=GemmaTextEncoderConfigurator,
        model_sd_ops=GEMMA_LLM_KEY_OPS,
        module_ops=(GEMMA_MODEL_OPS, *module_ops_from_gemma_root(str(gemma_model_path))),
    ).build(device=torch_device, dtype=dtype)


def load_embeddings_processor(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "EmbeddingsProcessor":
    from ltx_audio_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_audio_core.text_encoders.gemma import EMBEDDINGS_PROCESSOR_KEY_OPS, EmbeddingsProcessorConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=EmbeddingsProcessorConfigurator,
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
    ).build(device=_to_torch_device(device), dtype=dtype)


@dataclass
class AudioModelComponents:
    transformer: "LTXModel | None" = None
    audio_vae_encoder: "AudioEncoder | None" = None
    audio_vae_decoder: "AudioDecoder | None" = None
    vocoder: "Vocoder | None" = None
    text_conditioning: "TextConditioningComponents | None" = None
    scheduler: "LTX2Scheduler | None" = None

    @property
    def text_encoder(self) -> "GemmaTextEncoder | None":
        return self.text_conditioning.text_encoder if self.text_conditioning is not None else None

    @property
    def embeddings_processor(self) -> "EmbeddingsProcessor | None":
        return self.text_conditioning.embeddings_processor if self.text_conditioning is not None else None

    def unload_text_encoder(self, free_memory: bool = False) -> None:
        if self.text_conditioning is None:
            return
        self.text_conditioning.unload_text_encoder(free_memory=free_memory)


def _load_scheduler() -> "LTX2Scheduler":
    from ltx_audio_core.components.schedulers import LTX2Scheduler

    return LTX2Scheduler()


def _validate_checkpoint_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_text_conditioning_components(
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    *,
    with_text_encoder: bool = True,
    with_embeddings_processor: bool = True,
    load_text_encoder_in_8bit: bool = False,
) -> "TextConditioningComponents | None":
    if not with_text_encoder and not with_embeddings_processor:
        return None

    torch_device = _to_torch_device(device)

    text_encoder = None
    if with_text_encoder:
        if text_encoder_path is None:
            raise ValueError("text_encoder_path must be provided when with_text_encoder=True")
        text_encoder = load_text_encoder(
            text_encoder_path,
            torch_device,
            dtype,
            load_in_8bit=load_text_encoder_in_8bit,
        )

    embeddings_processor = None
    if with_embeddings_processor:
        embeddings_processor = load_embeddings_processor(checkpoint_path, torch_device, dtype)

    return TextConditioningComponents(
        text_encoder=text_encoder,
        embeddings_processor=embeddings_processor,
    )


def _load_components(  # noqa: PLR0913
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    *,
    with_transformer: bool = True,
    with_audio_vae_encoder: bool = False,
    with_audio_vae_decoder: bool = False,
    with_vocoder: bool = False,
    with_text_encoder: bool = False,
    with_embeddings_processor: bool = False,
    load_text_encoder_in_8bit: bool = False,
    with_scheduler: bool = True,
) -> AudioModelComponents:
    """Shared implementation for scene-specific component bundles."""
    checkpoint_path = _validate_checkpoint_path(checkpoint_path)
    torch_device = _to_torch_device(device)
    transformer = load_transformer(checkpoint_path, torch_device, dtype) if with_transformer else None
    audio_vae_encoder = load_audio_vae_encoder(checkpoint_path, torch_device, dtype) if with_audio_vae_encoder else None
    audio_vae_decoder = load_audio_vae_decoder(checkpoint_path, torch_device, dtype) if with_audio_vae_decoder else None
    vocoder = load_vocoder(checkpoint_path, torch_device, dtype) if with_vocoder else None
    text_conditioning = load_text_conditioning_components(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
        device=torch_device,
        dtype=dtype,
        with_text_encoder=with_text_encoder,
        with_embeddings_processor=with_embeddings_processor,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
    )

    return AudioModelComponents(
        transformer=transformer,
        audio_vae_encoder=audio_vae_encoder,
        audio_vae_decoder=audio_vae_decoder,
        vocoder=vocoder,
        text_conditioning=text_conditioning,
        scheduler=_load_scheduler() if with_scheduler else None,
    )


def load_training_components(
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    *,
    with_text_encoder: bool = True,
    with_embeddings_processor: bool = True,
    load_text_encoder_in_8bit: bool = False,
) -> AudioModelComponents:
    logger.info(f"Loading audio-only training components from {checkpoint_path}")
    return _load_components(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
        device=device,
        dtype=dtype,
        with_transformer=True,
        with_audio_vae_encoder=False,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=with_text_encoder,
        with_embeddings_processor=with_embeddings_processor,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        with_scheduler=True,
    )


def load_preprocess_components(
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    *,
    with_text_encoder: bool = True,
    with_embeddings_processor: bool = True,
    load_text_encoder_in_8bit: bool = False,
) -> AudioModelComponents:
    logger.info(f"Loading audio-only preprocessing components from {checkpoint_path}")
    return _load_components(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
        device=device,
        dtype=dtype,
        with_transformer=False,
        with_audio_vae_encoder=True,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=with_text_encoder,
        with_embeddings_processor=with_embeddings_processor,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        with_scheduler=False,
    )


def load_validation_components(
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    *,
    with_text_encoder: bool = True,
    with_embeddings_processor: bool = True,
    load_text_encoder_in_8bit: bool = False,
) -> AudioModelComponents:
    logger.info(f"Loading audio-only validation components from {checkpoint_path}")
    return _load_components(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
        device=device,
        dtype=dtype,
        with_transformer=True,
        with_audio_vae_encoder=False,
        with_audio_vae_decoder=True,
        with_vocoder=True,
        with_text_encoder=with_text_encoder,
        with_embeddings_processor=with_embeddings_processor,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        with_scheduler=True,
    )


def load_model(
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    with_audio_vae_encoder: bool = False,
    with_audio_vae_decoder: bool = True,
    with_vocoder: bool = True,
    with_text_encoder: bool = True,
    with_embeddings_processor: bool = True,
    load_text_encoder_in_8bit: bool = False,
) -> AudioModelComponents:
    logger.info(f"Loading audio-only LTX model from {checkpoint_path}")
    return _load_components(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
        device=device,
        dtype=dtype,
        with_transformer=True,
        with_audio_vae_encoder=with_audio_vae_encoder,
        with_audio_vae_decoder=with_audio_vae_decoder,
        with_vocoder=with_vocoder,
        with_text_encoder=with_text_encoder,
        with_embeddings_processor=with_embeddings_processor,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        with_scheduler=True,
    )
