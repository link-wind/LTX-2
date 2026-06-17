# ruff: noqa: PLC0415

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import torch

from ltx_audio_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
from ltx_audio_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer


def load_8bit_gemma(
    gemma_model_path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | int | None = None,
) -> GemmaTextEncoder:
    """Load the Gemma text encoder in 8-bit precision using bitsandbytes."""
    try:
        from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration
    except ImportError as e:
        raise ImportError(
            "8-bit text encoder loading requires bitsandbytes. Install it with: uv pip install bitsandbytes"
        ) from e

    gemma_path = _find_gemma_subpath(gemma_model_path, "model*.safetensors")
    tokenizer_path = _find_gemma_subpath(gemma_model_path, "tokenizer.model")

    device_map: str | dict[str, int | str | torch.device]
    if device is not None:
        device_map = {"": device}
    elif torch.cuda.is_available():
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    else:
        device_map = "auto"

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    with _suppress_accelerate_memory_warnings():
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            local_files_only=True,
        )

    tokenizer = LTXVGemmaTokenizer(tokenizer_path, 1024)
    return GemmaTextEncoder(
        tokenizer=tokenizer,
        model=gemma_model,
        dtype=dtype,
    )


def _find_gemma_subpath(root_path: str | Path, pattern: str) -> str:
    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return str(matches[0].parent)


@contextmanager
def _suppress_accelerate_memory_warnings() -> Generator[None, None, None]:
    accelerate_logger = logging.getLogger("accelerate.utils.modeling")
    old_level = accelerate_logger.level
    accelerate_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        accelerate_logger.setLevel(old_level)
