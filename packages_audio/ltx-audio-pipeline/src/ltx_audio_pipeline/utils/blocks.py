"""Pipeline blocks for audio-only inference.

These classes intentionally mirror the structure of ``ltx_pipelines.utils.blocks``
where possible so the package stays easy to merge back into the main pipeline
implementation later.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from typing import Callable, TypeVar

import torch

from ltx_audio_core.block_streaming import DISK_CPU_SLOTS, StreamingModelBuilder
from ltx_audio_core.components.diffusion_steps import EulerDiffusionStep
from ltx_audio_core.components.noisers import Noiser
from ltx_audio_core.components.patchifiers import AudioPatchifier
from ltx_audio_core.components.protocols import DiffusionStepProtocol
from ltx_audio_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_audio_core.loader.fuse_loras import bf16_fuse_rule
from ltx_audio_core.loader.attention_ops import set_attention_module_op
from ltx_audio_core.loader.module_ops import ModuleOps
from ltx_audio_core.loader.primitives import BuilderProtocol, LoraPathStrengthAndSDOps, ModelBuilderProtocol
from ltx_audio_core.loader.registry import DummyRegistry, Registry
from ltx_audio_core.loader.sd_ops import SDOps
from ltx_audio_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_audio_core.model.audio_vae.audio_vae import decode_audio as vae_decode_audio
from ltx_audio_core.model.audio_vae.model_configurator import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from ltx_audio_core.model.transformer import LTX_AUDIO_MODEL_COMFY_RENAMING_MAP, LTXModel, LTXModelConfigurator, Modality, X0Model
from ltx_audio_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_audio_core.model.transformer.compiling import (
    CompilationConfig,
    build_compile_transformer_op,
    modify_sd_ops_for_compilation,
)
from ltx_audio_core.quantization import QuantizationPolicy, fp8_cast_fuse_rule
from ltx_audio_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessor,
    EmbeddingsProcessorConfigurator,
    EmbeddingsProcessorOutput,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from ltx_audio_core.tools import AudioLatentTools, LatentTools
from ltx_audio_core.types import Audio, AudioLatentShape, LatentState, VideoPixelShape
from ltx_audio_core.utils import find_matching_file
from ltx_audio_pipeline.utils.gpu_model import gpu_model
from ltx_audio_pipeline.utils.helpers import cleanup_memory, create_noised_state, generate_enhanced_prompt
from ltx_audio_pipeline.utils.samplers import euler_denoising_loop
from ltx_audio_pipeline.utils.types import Denoiser, ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _chain_quantization(
    sd_ops: SDOps,
    module_ops: tuple[ModuleOps, ...],
    quantization: QuantizationPolicy,
) -> tuple[SDOps, tuple[ModuleOps, ...]]:
    chained_sd_ops = sd_ops
    if quantization.sd_ops is not None:
        chained_sd_ops = SDOps(
            name=f"sd_ops_chain_{sd_ops.name}+{quantization.sd_ops.name}",
            mapping=(*sd_ops.mapping, *quantization.sd_ops.mapping),
            allowed_keys=quantization.sd_ops.allowed_keys if sd_ops.allowed_keys is None else sd_ops.allowed_keys,
        )
    return chained_sd_ops, (*module_ops, *quantization.module_ops)


def _apply_compile_ops(
    sd_ops: SDOps,
    module_ops: tuple[ModuleOps, ...],
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    number_of_layers: int,
    compilation_config: CompilationConfig,
) -> tuple[SDOps, tuple[ModuleOps, ...], tuple[LoraPathStrengthAndSDOps, ...]]:
    sd_ops = modify_sd_ops_for_compilation(sd_ops, number_of_layers)
    compile_op = build_compile_transformer_op(compilation_config)
    module_ops = (*module_ops, compile_op)
    loras = tuple(
        LoraPathStrengthAndSDOps(
            lora.path,
            lora.strength,
            modify_sd_ops_for_compilation(lora.sd_ops, number_of_layers),
        )
        for lora in loras
    )
    return sd_ops, module_ops, loras


@contextmanager
def _streaming_model(
    builder: StreamingModelBuilder,
    offload_mode: OffloadMode,
    target_device: torch.device,
    dtype: torch.dtype,
) -> Iterator:
    cpu_slots_count = DISK_CPU_SLOTS if offload_mode == OffloadMode.DISK else None
    wrapped = builder.build(
        target_device=target_device,
        dtype=dtype,
        cpu_slots_count=cpu_slots_count,
    )
    try:
        yield wrapped
    finally:
        wrapped.teardown()
        wrapped.to("meta")
        cleanup_memory()


def _build_state(
    spec: ModalitySpec,
    tools: LatentTools,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
) -> LatentState:
    state = create_noised_state(
        tools=tools,
        conditionings=spec.conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=spec.noise_scale,
        initial_latent=spec.initial_latent,
    )
    if spec.frozen:
        state = replace(state, denoise_mask=torch.zeros_like(state.denoise_mask))
    return state


class _AudioOnlyAVAdapter(torch.nn.Module):
    """Expose an AV-style interface over the audio-only X0 model."""

    def __init__(self, model: X0Model):
        super().__init__()
        self.model = model

    def forward(
        self,
        video: Modality | None = None,
        audio: Modality | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[None, torch.Tensor]:
        if video is not None:
            raise ValueError("Audio-only transformer adapter does not support video inputs.")
        if audio is None:
            raise ValueError("Audio-only transformer adapter requires an audio modality.")
        return None, self.model(audio=audio, perturbations=perturbations)


class _AudioOnlyBatchSplitAdapter(torch.nn.Module):
    """Batch-split wrapper compatible with the AV-style denoiser interface."""

    def __init__(self, model: _AudioOnlyAVAdapter, max_batch_size: int = 1):
        super().__init__()
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.model = model
        self.max_batch_size = max_batch_size

    def forward(
        self,
        video: Modality | None = None,
        audio: Modality | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[None, torch.Tensor]:
        if video is not None:
            raise ValueError("Audio-only batch split adapter does not support video inputs.")
        if audio is None:
            raise ValueError("Audio-only batch split adapter requires an audio modality.")

        batch_size = audio.latent.shape[0]
        if batch_size <= self.max_batch_size:
            return self.model(video=video, audio=audio, perturbations=perturbations)

        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        chunk_sizes: list[int] = []
        remaining = batch_size
        while remaining > 0:
            chunk_size = min(self.max_batch_size, remaining)
            chunk_sizes.append(chunk_size)
            remaining -= chunk_size

        outputs: list[torch.Tensor] = []
        start = 0
        for chunk_size, audio_chunk in zip(chunk_sizes, audio.split(chunk_sizes), strict=True):
            stop = start + chunk_size
            perturbation_chunk = BatchedPerturbationConfig(perturbations.perturbations[start:stop])
            _, output = self.model(audio=audio_chunk, perturbations=perturbation_chunk)
            outputs.append(output)
            start = stop

        return None, torch.cat(outputs, dim=0)


class DiffusionStage:
    """Owns transformer lifecycle for the audio-only diffusion stage."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        transformer_builder: ModelBuilderProtocol[LTXModel] | None = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._quantization = quantization
        self._compilation_config = compilation_config
        self._offload_mode = offload_mode

        configurator = (
            quantization.model_configurator
            if quantization is not None and quantization.model_configurator is not None
            else LTXModelConfigurator
        )
        if transformer_builder is not None:
            self._transformer_builder = transformer_builder
        else:
            self._transformer_builder = Builder(
                model_path=checkpoint_path,
                model_class_configurator=configurator,
                model_sd_ops=LTX_AUDIO_MODEL_COMFY_RENAMING_MAP,
                loras=tuple(loras),
                registry=registry or DummyRegistry(),
            )

        if offload_mode != OffloadMode.NONE:
            if compilation_config is not None:
                raise ValueError("torch.compile is not supported with layer streaming")
            if quantization is not None and quantization.fuse_rule is not fp8_cast_fuse_rule:
                raise ValueError(
                    "Block streaming is not supported with this quantization policy "
                    "(only bf16 and fp8_cast are currently supported)."
                )
            streaming_sd_ops: SDOps = LTX_AUDIO_MODEL_COMFY_RENAMING_MAP
            streaming_module_ops: tuple[ModuleOps, ...] = ()
            if quantization is not None:
                streaming_sd_ops, streaming_module_ops = _chain_quantization(
                    streaming_sd_ops,
                    streaming_module_ops,
                    quantization,
                )
            self._streaming_builder = StreamingModelBuilder(
                model_class_configurator=configurator,
                model_path=checkpoint_path,
                model_sd_ops=streaming_sd_ops,
                module_ops=streaming_module_ops,
                loras=tuple(loras),
                registry=registry or DummyRegistry(),
                fuse_rule=quantization.fuse_rule if quantization is not None else bf16_fuse_rule,
                blocks_attr="transformer_blocks",
                blocks_prefix="transformer_blocks",
            )

    def with_attention(self, attention: AttentionFunction | AttentionCallable | None) -> "DiffusionStage":
        if attention is None:
            return self
        op = set_attention_module_op(attention)
        new = copy.copy(self)
        new._transformer_builder = self._transformer_builder.with_module_ops(
            (*self._transformer_builder.module_ops, op),
        )
        if self._offload_mode != OffloadMode.NONE:
            new._streaming_builder = dataclasses.replace(
                self._streaming_builder,
                module_ops=(*self._streaming_builder.module_ops, op),
            )
        return new

    def _build_transformer(self, *, device: torch.device | None = None, **kwargs: object) -> _AudioOnlyAVAdapter:
        target = device or self._device
        sd_ops = self._transformer_builder.model_sd_ops or LTX_AUDIO_MODEL_COMFY_RENAMING_MAP
        module_ops = self._transformer_builder.module_ops
        loras = self._transformer_builder.loras

        if self._compilation_config is not None:
            number_of_layers = self._transformer_builder.model_config()["transformer"]["num_layers"]
            sd_ops, module_ops, loras = _apply_compile_ops(
                sd_ops,
                module_ops,
                loras,
                number_of_layers,
                self._compilation_config,
            )

        if self._quantization is not None:
            sd_ops, module_ops = _chain_quantization(sd_ops, module_ops, self._quantization)

        builder = self._transformer_builder.with_module_ops(module_ops).with_sd_ops(sd_ops).with_loras(loras)
        if self._quantization is not None:
            builder = builder.with_fuse_rule(self._quantization.fuse_rule)
        model = X0Model(builder.build(device=target, **kwargs)).to(target).eval()
        return _AudioOnlyAVAdapter(model).to(target).eval()

    @contextmanager
    def _streaming_transformer_ctx(self) -> Iterator[_AudioOnlyAVAdapter]:
        with _streaming_model(
            self._streaming_builder,
            self._offload_mode,
            self._device,
            self._dtype,
        ) as streaming_wrapper:
            yield _AudioOnlyAVAdapter(X0Model(streaming_wrapper).eval())

    def _transformer_ctx(self, **kwargs: object) -> AbstractContextManager:
        if self._offload_mode != OffloadMode.NONE:
            return self._streaming_transformer_ctx()
        return gpu_model(self._build_transformer(**kwargs))

    def model_context(self, **kwargs: object) -> AbstractContextManager:
        return self._transformer_ctx(**kwargs)

    def run(  # noqa: PLR0913
        self,
        transformer: object,
        denoiser: Denoiser,
        sigmas: torch.Tensor,
        noiser: Noiser,
        width: int,
        height: int,
        frames: int,
        fps: float,
        video: ModalitySpec | None = None,
        audio: ModalitySpec | None = None,
        stepper: DiffusionStepProtocol | None = None,
        loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
        max_batch_size: int = 1,
    ) -> tuple[None, LatentState | None]:
        if video is not None:
            raise ValueError("Audio-only DiffusionStage currently requires video=None.")
        if audio is None:
            raise ValueError("Audio-only DiffusionStage requires an audio ModalitySpec.")

        if loop is None:
            loop = euler_denoising_loop
        if stepper is None:
            stepper = EulerDiffusionStep()

        pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)
        audio_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), audio_shape)
        audio_state = _build_state(audio, audio_tools, noiser, self._dtype, self._device)

        wrapped = _AudioOnlyBatchSplitAdapter(transformer, max_batch_size=max_batch_size)  # type: ignore[arg-type]
        _, audio_state = loop(
            sigmas=sigmas,
            video_state=None,
            audio_state=audio_state,
            stepper=stepper,
            transformer=wrapped,
            denoiser=denoiser,
        )

        if audio_state is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)

        return None, audio_state

    def __call__(  # noqa: PLR0913
        self,
        denoiser: Denoiser,
        sigmas: torch.Tensor,
        noiser: Noiser,
        width: int,
        height: int,
        frames: int,
        fps: float,
        video: ModalitySpec | None = None,
        audio: ModalitySpec | None = None,
        stepper: DiffusionStepProtocol | None = None,
        loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
        max_batch_size: int = 1,
    ) -> tuple[None, LatentState | None]:
        mode = "streaming" if self._offload_mode != OffloadMode.NONE else "standard"
        logger.info("Building transformer (%s) from %s", mode, self._checkpoint_path)
        with self._transformer_ctx() as transformer:
            logger.info("Running audio denoising loop (%d steps, %d frames @ %.1f fps)", len(sigmas) - 1, frames, fps)
            return self.run(
                transformer,
                denoiser,
                sigmas,
                noiser,
                width,
                height,
                frames,
                fps,
                video,
                audio,
                stepper,
                loop,
                max_batch_size,
            )


class PromptEncoder:
    """Owns text encoder + embeddings processor lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        text_encoder_builder: BuilderProtocol | None = None,
    ) -> None:
        self._gemma_root = gemma_root
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._offload_mode = offload_mode

        if text_encoder_builder is not None:
            if offload_mode != OffloadMode.NONE:
                raise ValueError(
                    "text_encoder_builder cannot be used with offload_mode != OffloadMode.NONE "
                    "because no streaming text encoder builder is available."
                )
            self._text_encoder_builder = text_encoder_builder
            self._streaming_text_encoder_builder = None
        else:
            module_ops = module_ops_from_gemma_root(gemma_root)
            model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
            weight_paths = [str(path) for path in model_folder.rglob("*.safetensors")]
            self._text_encoder_builder = Builder(
                model_path=tuple(weight_paths),
                model_class_configurator=GemmaTextEncoderConfigurator,
                model_sd_ops=GEMMA_LLM_KEY_OPS,
                module_ops=(GEMMA_MODEL_OPS, *module_ops),
                registry=registry or DummyRegistry(),
            )
            self._streaming_text_encoder_builder = StreamingModelBuilder(
                model_path=tuple(weight_paths),
                model_class_configurator=GemmaTextEncoderConfigurator,
                model_sd_ops=GEMMA_LLM_KEY_OPS,
                module_ops=(GEMMA_MODEL_OPS, *module_ops),
                registry=registry or DummyRegistry(),
                blocks_attr="model.model.language_model.layers",
                blocks_prefix="model.model.language_model.layers",
            )
        self._embeddings_processor_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=registry or DummyRegistry(),
        )

    def _build_text_encoder(self) -> torch.nn.Module:
        return self._text_encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def _build_embeddings_processor(self) -> EmbeddingsProcessor:
        return self._embeddings_processor_builder.build(device=self._device, dtype=self._dtype).eval()

    def _text_encoder_ctx(self) -> AbstractContextManager:
        if self._offload_mode != OffloadMode.NONE:
            return _streaming_model(
                self._streaming_text_encoder_builder,
                self._offload_mode,
                self._device,
                self._dtype,
            )
        return gpu_model(self._build_text_encoder())

    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
    ) -> list[EmbeddingsProcessorOutput]:
        logger.info("Building text encoder from %s", self._gemma_root)
        with self._text_encoder_ctx() as text_encoder:
            if enhance_first_prompt:
                prompts = list(prompts)
                prompts[0] = generate_enhanced_prompt(
                    text_encoder,
                    prompts[0],
                    enhance_prompt_image,
                    seed=enhance_prompt_seed,
                )
            raw_outputs = [text_encoder.encode(prompt) for prompt in prompts]
        logger.info("Text encoder done, building embeddings processor from %s", self._checkpoint_path)
        with gpu_model(self._build_embeddings_processor()) as embeddings_processor:
            result = [embeddings_processor.process_hidden_states(hidden_states, mask) for hidden_states, mask in raw_outputs]
        logger.info("Prompt encoding complete")
        return result


class AudioDecoder:
    """Owns audio decoder + vocoder lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._decoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._vocoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def __call__(self, latent: torch.Tensor) -> Audio:
        logger.info("Building audio decoder + vocoder from %s", self._checkpoint_path)
        with (
            gpu_model(self._decoder_builder.build(device=self._device, dtype=self._dtype).eval()) as decoder,
            gpu_model(self._vocoder_builder.build(device=self._device, dtype=self._dtype).eval()) as vocoder,
        ):
            return vae_decode_audio(latent, decoder, vocoder)


class AudioConditioner:
    """Owns audio encoder lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def __call__(self, fn: Callable[[torch.nn.Module], T]) -> T:
        with gpu_model(self._encoder_builder.build(device=self._device, dtype=self._dtype).eval()) as encoder:
            return fn(encoder)


__all__ = [
    "AudioConditioner",
    "AudioDecoder",
    "DiffusionStage",
    "PromptEncoder",
]
