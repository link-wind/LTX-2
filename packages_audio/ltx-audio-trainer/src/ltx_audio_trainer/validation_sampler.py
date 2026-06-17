from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from ltx_audio_core.components.diffusion_steps import EulerDiffusionStep
from ltx_audio_core.components.noisers import GaussianNoiser
from ltx_audio_core.components.patchifiers import AudioPatchifier
from ltx_audio_core.components.schedulers import LTX2Scheduler
from ltx_audio_core.model.audio_vae.audio_vae import decode_audio
from ltx_audio_core.model.transformer.modality import Modality
from ltx_audio_core.model.transformer.model import X0Model
from ltx_audio_core.tools import AudioLatentTools
from ltx_audio_core.types import Audio, AudioLatentShape
from ltx_audio_trainer.progress import SamplingContext

if TYPE_CHECKING:
    from ltx_audio_core.model.audio_vae.audio_vae import AudioDecoder
    from ltx_audio_core.model.audio_vae.vocoder import Vocoder
    from ltx_audio_core.model.transformer.model import LTXModel
    from ltx_audio_core.text_encoders.gemma import EmbeddingsProcessor, GemmaTextEncoder


@dataclass
class CachedPromptEmbeddings:
    audio_context_positive: torch.Tensor
    audio_context_negative: torch.Tensor | None
    audio_attention_mask_positive: torch.Tensor
    audio_attention_mask_negative: torch.Tensor | None


@dataclass
class GenerationConfig:
    prompt: str
    negative_prompt: str = ""
    audio_duration_seconds: float = 10.0
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    seed: int = 42
    cached_embeddings: CachedPromptEmbeddings | None = None


class ValidationSampler:
    def __init__(
        self,
        transformer: "LTXModel | torch.nn.Module",
        audio_decoder: "AudioDecoder",
        vocoder: "Vocoder",
        text_encoder: "GemmaTextEncoder | None" = None,
        embeddings_processor: "EmbeddingsProcessor | None" = None,
        sampling_context: SamplingContext | None = None,
        scheduler: LTX2Scheduler | None = None,
    ) -> None:
        self._x0_model = transformer if isinstance(transformer, X0Model) else X0Model(transformer)
        self._audio_decoder = audio_decoder
        self._vocoder = vocoder
        self._text_encoder = text_encoder
        self._embeddings_processor = embeddings_processor
        self._sampling_context = sampling_context
        self._scheduler = scheduler or LTX2Scheduler()
        self._audio_patchifier = AudioPatchifier(patch_size=1)

    @torch.no_grad()
    def generate(
        self,
        config: GenerationConfig,
        device: torch.device | str = "cuda",
    ) -> Audio:
        device = torch.device(device) if isinstance(device, str) else device
        pos_context, pos_mask, neg_context, neg_mask = self._get_prompt_embeddings(config, device)
        if hasattr(self._x0_model, "to"):
            self._x0_model.to(device)
        if hasattr(self._audio_decoder, "to"):
            self._audio_decoder.to(device)
        if hasattr(self._vocoder, "to"):
            self._vocoder.to(device)

        generator = torch.Generator(device=device).manual_seed(config.seed)
        noiser = GaussianNoiser(generator=generator)
        audio_tools = AudioLatentTools(
            self._audio_patchifier,
            AudioLatentShape.from_duration(batch=1, duration=config.audio_duration_seconds),
        )
        audio_state = audio_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        audio_state = noiser(audio_state, noise_scale=1.0)

        sigmas = self._scheduler.execute(steps=config.num_inference_steps).to(device=device, dtype=torch.float32)
        stepper = EulerDiffusionStep()

        for step_index, sigma in enumerate(sigmas[:-1]):
            timestep_value = sigma.repeat(audio_state.latent.shape[0])
            timesteps = timestep_value.view(-1, 1, 1).expand(-1, audio_state.latent.shape[1], 1)
            positive = Modality(
                latent=audio_state.latent,
                sigma=timestep_value,
                timesteps=timesteps,
                positions=audio_state.positions.to(device=device, dtype=audio_state.latent.dtype),
                context=pos_context,
                context_mask=pos_mask,
            )
            denoised = self._x0_model(audio=positive, perturbations=None)

            if config.guidance_scale != 1.0 and neg_context is not None and neg_mask is not None:
                negative = Modality(
                    latent=audio_state.latent,
                    sigma=timestep_value,
                    timesteps=timesteps,
                    positions=audio_state.positions.to(device=device, dtype=audio_state.latent.dtype),
                    context=neg_context,
                    context_mask=neg_mask,
                )
                uncond = self._x0_model(audio=negative, perturbations=None)
                denoised = uncond + config.guidance_scale * (denoised - uncond)

            next_latent = stepper.step(
                sample=audio_state.latent,
                denoised_sample=denoised,
                sigmas=sigmas,
                step_index=step_index,
            )
            audio_state = replace(audio_state, latent=next_latent)
            if self._sampling_context is not None:
                self._sampling_context.advance_step()

        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)
        return decode_audio(audio_state.latent, self._audio_decoder, self._vocoder)

    def _get_prompt_embeddings(
        self,
        config: GenerationConfig,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if config.cached_embeddings is not None:
            cached = config.cached_embeddings
            pos_context = cached.audio_context_positive.to(device)
            pos_mask = cached.audio_attention_mask_positive.to(device)
            neg_context = (
                cached.audio_context_negative.to(device) if cached.audio_context_negative is not None else None
            )
            neg_mask = (
                cached.audio_attention_mask_negative.to(device)
                if cached.audio_attention_mask_negative is not None
                else None
            )
            return pos_context, pos_mask, neg_context, neg_mask

        if self._text_encoder is None or self._embeddings_processor is None:
            raise ValueError("Either cached_embeddings or text encoder + embeddings processor must be provided")

        return self._encode_prompts(config, device)

    def _encode_prompts(
        self,
        config: GenerationConfig,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if hasattr(self._text_encoder, "to"):
            self._text_encoder.to(device)
        elif hasattr(self._text_encoder, "model") and hasattr(self._text_encoder.model, "to"):
            self._text_encoder.model.to(device)

        if hasattr(self._embeddings_processor, "to"):
            self._embeddings_processor.to(device)

        pos_hs, pos_mask = self._text_encoder.encode(config.prompt)
        pos_out = self._embeddings_processor.process_hidden_states(pos_hs, pos_mask)
        pos_context = pos_out.audio_encoding if pos_out.audio_encoding is not None else pos_out.video_encoding
        pos_attention_mask = pos_out.attention_mask

        neg_context = None
        neg_attention_mask = None
        if config.guidance_scale != 1.0:
            neg_hs, neg_mask = self._text_encoder.encode(config.negative_prompt)
            neg_out = self._embeddings_processor.process_hidden_states(neg_hs, neg_mask)
            neg_context = neg_out.audio_encoding if neg_out.audio_encoding is not None else neg_out.video_encoding
            neg_attention_mask = neg_out.attention_mask

        if hasattr(self._text_encoder, "model") and hasattr(self._text_encoder.model, "to"):
            self._text_encoder.model.to("cpu")

        return (
            pos_context.to(device),
            pos_attention_mask.to(device),
            neg_context.to(device) if neg_context is not None else None,
            neg_attention_mask.to(device) if neg_attention_mask is not None else None,
        )
