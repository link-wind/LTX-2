"""High-level one-stage audio pipeline.

This module intentionally mirrors the structure of
``ltx_pipelines.ti2vid_one_stage`` where practical, while keeping the
conditioning surface audio-only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchaudio

from ltx_audio_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_audio_core.components.noisers import GaussianNoiser
from ltx_audio_core.components.schedulers import LTX2Scheduler
from ltx_audio_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_audio_core.loader.registry import Registry
from ltx_audio_core.model.transformer.compiling import CompilationConfig
from ltx_audio_core.quantization import QuantizationPolicy
from ltx_audio_core.types import Audio
from ltx_audio_pipeline.utils import (
    AudioDecoder,
    DiffusionStage,
    FactoryGuidedDenoiser,
    ModalitySpec,
    OffloadMode,
    PromptEncoder,
    SimpleDenoiser,
    get_device,
)
from ltx_audio_pipeline.utils.args import audio_one_stage_arg_parser, detect_checkpoint_path
from ltx_audio_pipeline.utils.constants import detect_params

logger = logging.getLogger(__name__)


class AudioOneStagePipeline:
    """Single-stage prompt-to-audio generation pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
    ) -> None:
        self.dtype = torch.bfloat16
        self.device = device or get_device()
        self._scheduler = LTX2Scheduler()
        self.prompt_encoder = PromptEncoder(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            offload_mode=offload_mode,
        )
        self.stage = DiffusionStage(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory | None = None,
        enhance_prompt: bool = False,
        max_batch_size: int = 1,
        sigmas: torch.Tensor | None = None,
    ) -> Audio:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=None,
            enhance_prompt_seed=seed,
        )
        audio_context = ctx_p.audio_encoding
        negative_audio_context = ctx_n.audio_encoding
        if audio_context is None:
            raise RuntimeError("PromptEncoder did not return audio conditioning for the positive prompt.")

        sigmas = (sigmas if sigmas is not None else self._scheduler.execute(steps=num_inference_steps)).to(
            dtype=torch.float32,
            device=self.device,
        )

        audio = ModalitySpec(context=audio_context)
        if audio_guider_params is None:
            denoiser = SimpleDenoiser(v_context=None, a_context=audio.context)
        else:
            audio_guider_factory = create_multimodal_guider_factory(
                params=audio_guider_params,
                negative_context=negative_audio_context,
            )
            denoiser = FactoryGuidedDenoiser(
                v_context=None,
                a_context=audio.context,
                audio_guider_factory=audio_guider_factory,
            )

        _, audio_state = self.stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=None,
            audio=audio,
            max_batch_size=max_batch_size,
        )
        if audio_state is None:
            raise RuntimeError("Audio diffusion stage returned no audio latent state.")

        return self.audio_decoder(audio_state.latent)


def _save_audio(audio: Audio, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    waveform = audio.waveform.detach().cpu()
    if waveform.dim() == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(str(path), waveform, audio.sampling_rate)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = audio_one_stage_arg_parser(params=params)
    args = parser.parse_args()

    pipeline = AudioOneStagePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
    )
    audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        enhance_prompt=args.enhance_prompt,
        max_batch_size=args.max_batch_size,
    )
    _save_audio(audio, args.output_path)
    logger.info("Saved audio to %s", args.output_path)


__all__ = [
    "AudioOneStagePipeline",
]


if __name__ == "__main__":
    main()
