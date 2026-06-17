from __future__ import annotations

import gc
import logging

import torch

from ltx_audio_core.components.noisers import Noiser
from ltx_audio_core.conditioning import ConditioningItem
from ltx_audio_core.model.transformer import Modality
from ltx_audio_core.text_encoders.gemma import GemmaTextEncoder
from ltx_audio_core.tools import LatentTools
from ltx_audio_core.types import LatentState


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        logging.warning("Host empty cache cleanup failed; ignoring.", exc_info=True)


def create_noised_state(
    tools: LatentTools,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> LatentState:
    """Create a noised latent state from empty state, conditionings, and noiser."""
    state = tools.create_initial_state(device, dtype, initial_latent)
    state = state_with_conditionings(state, conditionings, tools)
    state = noiser(state, noise_scale)
    return state


def state_with_conditionings(
    latent_state: LatentState,
    conditioning_items: list[ConditioningItem],
    latent_tools: LatentTools,
) -> LatentState:
    """Apply a list of conditionings to a latent state."""
    for conditioning in conditioning_items:
        latent_state = conditioning.apply_to(latent_state=latent_state, latent_tools=latent_tools)
    return latent_state


def post_process_latent(denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Blend denoised output with clean state based on mask."""
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def modality_from_latent_state(
    state: LatentState,
    context: torch.Tensor,
    sigma: torch.Tensor,
    enabled: bool = True,
) -> Modality:
    """Create a Modality from a latent state."""
    return Modality(
        enabled=enabled,
        latent=state.latent,
        sigma=sigma,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=state.attention_mask,
    )


def timesteps_from_mask(denoise_mask: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """Compute per-token timesteps from a denoise mask and sigma value."""
    if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
        sigma = sigma.view(-1, *([1] * (denoise_mask.dim() - 1)))
    return denoise_mask * sigma


_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")


def clean_response(text: str) -> str:
    """Normalize Gemma-style punctuation and trim leading non-letter characters."""
    text = text.translate(_UNICODE_REPLACEMENTS)
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def generate_enhanced_prompt(
    text_encoder: GemmaTextEncoder,
    prompt: str,
    image_path: str | None = None,
    image_long_side: int = 896,  # noqa: ARG001
    seed: int = 42,
) -> str:
    """Generate a cleaned prompt enhancement via the text encoder.

    The audio-only pipeline currently supports text-only prompt enhancement.
    Image-conditioned prompt enhancement can be added later if the audio
    package grows an image preprocessing path mirroring the AV pipeline.
    """
    if image_path is not None:
        raise ValueError("image-conditioned prompt enhancement is not yet supported in ltx-audio-pipeline.")

    prompt = text_encoder.enhance_t2v(prompt, seed=seed)
    logging.info("Enhanced prompt: %s", prompt)
    return clean_response(prompt)
