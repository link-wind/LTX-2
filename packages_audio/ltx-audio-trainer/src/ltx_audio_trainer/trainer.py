from __future__ import annotations

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torchaudio
import wandb
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DistributedType, gather_object, set_seed
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader

from ltx_audio_trainer import logger
from ltx_audio_trainer.config import AudioTrainerConfig
from ltx_audio_trainer.config_display import print_config
from ltx_audio_trainer.datasets import PrecomputedDataset
from ltx_audio_trainer.gpu_utils import get_gpu_memory_gb
from ltx_audio_trainer.hf_hub_utils import push_to_hub
from ltx_audio_trainer.model_loader import (
    load_audio_vae_decoder,
    load_text_conditioning_components,
    load_training_components,
    load_vocoder,
)
from ltx_audio_trainer.progress import TrainingProgress
from ltx_audio_trainer.sigma_tracker import SigmaBucketTracker
from ltx_audio_trainer.timestep_samplers import SAMPLERS
from ltx_audio_trainer.training_state import ConfigFingerprint, RngStates, TrainingState
from ltx_audio_trainer.training_strategies import get_training_strategy
from ltx_audio_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler

if TYPE_CHECKING:
    from ltx_audio_core.types import Audio

MEMORY_CHECK_INTERVAL = 100


@dataclass
class TrainingStats:
    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


@dataclass(frozen=True)
class TrainingStepOutput:
    loss: Tensor
    sigma: Tensor


class AudioTrainer:
    def __init__(self, config: AudioTrainerConfig) -> None:
        self._config = config
        print_config(config)
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        self._setup_accelerator()
        self._load_models()
        self._collect_trainable_params()
        self._dataset: Any = None
        self._dataloader: Any = None
        self._optimizer: Any = None
        self._lr_scheduler: Any = None
        self._timestep_sampler: Any = None
        self._global_step = -1
        self._sigma_tracker = SigmaBucketTracker()
        self._loaded_checkpoint_path: Path | None = None
        self._resume_state: tuple[int, TrainingState | None] = (0, None)
        self._checkpoint_paths: list[Path] = []
        self._training_state_paths: list[Path] = []
        self._training_state_size_warned = False
        self._model_prepared = False
        self._audio_decoder = None
        self._vocoder = None
        self._wandb_run = None
        self._cached_validation_embeddings = self._cache_validation_embeddings()
        self._load_checkpoint()

    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
    ) -> tuple[Path, TrainingStats]:
        set_seed(self._config.seed)
        self._prepare_model_for_training()
        self._init_optimizer()
        self._init_dataloader()
        self._init_timestep_sampler()

        initial_step, training_state = self._resume_state
        resuming = training_state is not None
        if training_state is not None and not self._restore_training_state(training_state):
            initial_step, training_state = 0, None
            self._resume_state = (0, None)
            resuming = False
        resume_run_id = training_state.wandb_run_id if resuming and training_state is not None else None
        self._init_wandb(resume_run_id=resume_run_id)

        remaining_steps = self._config.optimization.steps - initial_step
        if remaining_steps <= 0:
            raise ValueError("No remaining training steps to run")

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

        progress = TrainingProgress(
            enabled=self._accelerator.is_local_main_process and not disable_progress_bars,
            total_steps=remaining_steps,
        )
        self._transformer.train()
        self._global_step = initial_step
        train_start_time = time.time()
        peak_mem = get_gpu_memory_gb(self._accelerator.device)

        data_iter = iter(self._dataloader)
        with progress:
            if (
                self._config.validation.interval
                and self._config.validation.prompts
                and not self._config.validation.skip_initial_validation
            ):
                with self._offloaded_optimizer_state():
                    self._run_distributed_validation(progress)

            for raw_step in range(remaining_steps * self._config.optimization.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                step_start = time.time()
                with self._accelerator.accumulate(self._transformer):
                    output = self._training_step(batch)
                    self._accelerator.backward(output.loss.mean())

                    if (
                        self._accelerator.sync_gradients
                        and self._config.optimization.max_grad_norm > 0
                        and self._trainable_params
                    ):
                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            self._config.optimization.max_grad_norm,
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    is_optimization_step = (
                        (raw_step + 1) % self._config.optimization.gradient_accumulation_steps == 0
                    )
                    if is_optimization_step:
                        self._global_step += 1
                        step_loss = output.loss.detach().mean().item()
                        current_lr = self._optimizer.param_groups[0]["lr"]
                        step_time = (
                            time.time() - step_start
                        ) * self._config.optimization.gradient_accumulation_steps
                        progress.update_training(
                            loss=step_loss,
                            lr=current_lr,
                            step_time=step_time,
                            advance=True,
                        )
                        self._sigma_tracker.update(
                            output.sigma.detach().cpu().tolist(),
                            output.loss.detach().cpu().tolist(),
                        )
                        self._log_metrics(
                            {
                                "train/loss": step_loss,
                                "train/learning_rate": current_lr,
                                "train/step_time": step_time,
                                "train/global_step": self._global_step,
                                **self._sigma_tracker.get_metrics(),
                            }
                        )

                        if (
                            self._config.validation.interval
                            and self._config.validation.prompts
                            and self._global_step > 0
                            and self._global_step % self._config.validation.interval == 0
                        ):
                            with self._offloaded_optimizer_state():
                                self._run_distributed_validation(progress)

                        if (
                            self._config.checkpoints.interval
                            and self._global_step > 0
                            and self._global_step % self._config.checkpoints.interval == 0
                        ):
                            self._save_checkpoint()

                if raw_step % MEMORY_CHECK_INTERVAL == 0:
                    peak_mem = max(peak_mem, get_gpu_memory_gb(self._accelerator.device))

        saved_path = self._save_checkpoint()
        total_time_seconds = time.time() - train_start_time
        steps_per_second = remaining_steps / max(total_time_seconds, 1e-8)
        samples_per_second = steps_per_second * self._config.optimization.batch_size * self._accelerator.num_processes
        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=self._config.optimization.batch_size * self._accelerator.num_processes,
        )
        if self._config.hub.push_to_hub:
            push_to_hub(saved_path, None, self._config)
        if self._wandb_run is not None:
            self._log_metrics(
                {
                    "stats/total_time_minutes": stats.total_time_seconds / 60,
                    "stats/steps_per_second": stats.steps_per_second,
                    "stats/samples_per_second": stats.samples_per_second,
                    "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                }
            )
            self._wandb_run.finish()
        if hasattr(self._accelerator, "end_training"):
            self._accelerator.end_training()
        return saved_path, stats

    def _setup_accelerator(self) -> None:
        self._accelerator = Accelerator(
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        )

    def _load_models(self) -> None:
        components = load_training_components(
            checkpoint_path=self._config.model.model_path,
            text_encoder_path=self._config.model.text_encoder_path,
            device="cpu",
            with_text_encoder=False,
            with_embeddings_processor=False,
            load_text_encoder_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        )
        if components.transformer is None:
            raise ValueError("Training components did not include a transformer")
        self._transformer = components.transformer
        self._scheduler = components.scheduler

    def _collect_trainable_params(self) -> None:
        if self._config.model.training_mode == "full":
            if hasattr(self._transformer, "requires_grad_"):
                self._transformer.requires_grad_(True)
        elif self._config.model.training_mode == "lora":
            self._setup_lora()
        else:
            raise ValueError(f"Unsupported training mode: {self._config.model.training_mode}")

        self._trainable_params = [param for param in self._transformer.parameters() if param.requires_grad]
        if not self._trainable_params:
            raise ValueError("No trainable parameters were collected")

    def _setup_lora(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise ImportError("LoRA training requires the `peft` package to be installed") from exc

        if self._config.lora is None:
            raise ValueError("LoRA config is required when training_mode='lora'")

        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        self._transformer = get_peft_model(self._transformer, lora_config)

    def _prepare_model_for_training(self) -> None:
        if self._model_prepared or not isinstance(self._transformer, torch.nn.Module):
            return

        if (
            getattr(self._accelerator, "distributed_type", None) == DistributedType.FSDP
            and self._config.model.training_mode == "lora"
        ):
            self._transformer = self._transformer.to(dtype=torch.float32)

        transformer = (
            self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
        )
        if hasattr(transformer, "set_gradient_checkpointing"):
            transformer.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)

        self._transformer = self._accelerator.prepare(self._transformer)
        self._model_prepared = True

    def _init_optimizer(self) -> None:
        optimizer_type = self._config.optimization.optimizer_type
        lr = self._config.optimization.learning_rate

        if optimizer_type == "adamw":
            optimizer = AdamW(self._trainable_params, lr=lr)
        elif optimizer_type == "adamw8bit":
            from bitsandbytes.optim import AdamW8bit  # noqa: PLC0415

            optimizer = AdamW8bit(self._trainable_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        lr_scheduler = self._create_scheduler(optimizer)
        self._optimizer, self._lr_scheduler = self._accelerator.prepare(optimizer, lr_scheduler)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        scheduler_type = self._config.optimization.scheduler_type
        params = self._config.optimization.scheduler_params
        total_steps = max(1, self._config.optimization.steps)

        if scheduler_type == "constant":
            return None
        if scheduler_type == "linear":
            return LinearLR(
                optimizer,
                start_factor=params.get("start_factor", 1.0),
                end_factor=params.get("end_factor", 0.0),
                total_iters=params.get("total_iters", total_steps),
            )
        if scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=params.get("t_max", total_steps))
        if scheduler_type == "cosine_with_restarts":
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.get("t_0", max(1, total_steps // 4)),
                T_mult=params.get("t_mult", 1),
            )
        if scheduler_type == "polynomial":
            return PolynomialLR(
                optimizer,
                total_iters=params.get("total_iters", total_steps),
                power=params.get("power", 1.0),
            )
        if scheduler_type == "step":
            return StepLR(
                optimizer,
                step_size=params.get("step_size", max(1, total_steps // 3)),
                gamma=params.get("gamma", 0.1),
            )
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    @contextmanager
    def _offloaded_optimizer_state(self) -> Iterator[None]:
        enabled = (
            getattr(self._config.acceleration, "offload_optimizer_during_validation", False)
            and self._optimizer is not None
            and getattr(self._accelerator, "distributed_type", None) != DistributedType.FSDP
        )

        offloaded: list[tuple[dict[str, object], str]] = []
        if enabled:
            for state in self._optimizer.state.values():
                for key, value in state.items():
                    if getattr(value, "is_cuda", False) and hasattr(value, "cpu") and hasattr(value, "to"):
                        offloaded.append((state, key))
            for state, key in offloaded:
                state[key] = state[key].cpu()

        try:
            yield
        finally:
            device = getattr(self._accelerator, "device", None)
            if device is not None:
                for state, key in offloaded:
                    state[key] = state[key].to(device)

    def _init_dataloader(self) -> None:
        if self._dataset is None:
            data_sources = self._training_strategy.get_data_sources()
            self._dataset = PrecomputedDataset(
                self._config.data.preprocessed_data_root,
                data_sources=data_sources,
            )

        num_workers = self._config.data.num_dataloader_workers
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )
        self._dataloader = self._accelerator.prepare(dataloader)

    def _init_timestep_sampler(self) -> None:
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> TrainingStepOutput:
        model_inputs = self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler)
        audio_pred = self._transformer(model_inputs.audio)
        loss = self._training_strategy.compute_loss(audio_pred, model_inputs)
        return TrainingStepOutput(loss=loss, sigma=model_inputs.audio.sigma.detach())

    def _cache_validation_embeddings(self) -> list[CachedPromptEmbeddings] | None:
        prompts = self._config.validation.prompts
        if not prompts:
            return None
        if self._config.model.text_encoder_path is None:
            raise ValueError("Validation prompts require model.text_encoder_path to be configured")

        text_stack = load_text_conditioning_components(
            checkpoint_path=self._config.model.model_path,
            text_encoder_path=self._config.model.text_encoder_path,
            device="cpu",
            with_text_encoder=True,
            with_embeddings_processor=True,
            load_text_encoder_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        )
        if text_stack is None:
            raise ValueError("Failed to load text conditioning stack for validation caching")

        text_encoder = text_stack.require_text_encoder()
        embeddings_processor = text_stack.require_embeddings_processor()
        cached_embeddings: list[CachedPromptEmbeddings] = []

        with torch.inference_mode():
            for prompt in prompts:
                pos_hs, pos_mask = text_encoder.encode(prompt)
                pos_out = embeddings_processor.process_hidden_states(pos_hs, pos_mask)
                neg_hs, neg_mask = text_encoder.encode(self._config.validation.negative_prompt)
                neg_out = embeddings_processor.process_hidden_states(neg_hs, neg_mask)

                pos_audio = pos_out.audio_encoding if pos_out.audio_encoding is not None else pos_out.video_encoding
                neg_audio = neg_out.audio_encoding if neg_out.audio_encoding is not None else neg_out.video_encoding
                cached_embeddings.append(
                    CachedPromptEmbeddings(
                        audio_context_positive=pos_audio.cpu(),
                        audio_context_negative=neg_audio.cpu(),
                        audio_attention_mask_positive=pos_out.attention_mask.cpu(),
                        audio_attention_mask_negative=neg_out.attention_mask.cpu(),
                    )
                )

        text_stack.unload_text_encoder(free_memory=False)
        return cached_embeddings

    def _ensure_validation_components(self) -> None:
        if self._audio_decoder is None:
            self._audio_decoder = load_audio_vae_decoder(
                self._config.model.model_path,
                device=self._accelerator.device,
            )
        if self._vocoder is None:
            self._vocoder = load_vocoder(
                self._config.model.model_path,
                device=self._accelerator.device,
            )

    def _run_validation(self, progress: TrainingProgress) -> list[Path]:
        if not self._config.validation.prompts:
            return []
        if not self._accelerator.is_local_main_process:
            self._accelerator.wait_for_everyone()
            return []

        self._ensure_validation_components()
        sampling_ctx = progress.start_sampling(
            num_prompts=len(self._config.validation.prompts),
            num_steps=self._config.validation.inference_steps,
        )
        transformer = (
            self._accelerator.unwrap_model(self._transformer)
            if hasattr(self._accelerator, "unwrap_model")
            else self._transformer
        )
        sampler = ValidationSampler(
            transformer=transformer,
            audio_decoder=self._audio_decoder,
            vocoder=self._vocoder,
            sampling_context=sampling_ctx,
            scheduler=self._scheduler,
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: list[Path] = []
        step_value = max(self._global_step, 0)

        for prompt_idx, prompt in enumerate(self._config.validation.prompts):
            sampling_ctx.start_video(prompt_idx)
            gen_config = GenerationConfig(
                prompt=prompt,
                negative_prompt=self._config.validation.negative_prompt,
                audio_duration_seconds=self._config.validation.audio_duration_seconds,
                num_inference_steps=self._config.validation.inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                seed=self._config.validation.seed,
                cached_embeddings=(
                    self._cached_validation_embeddings[prompt_idx]
                    if self._cached_validation_embeddings is not None
                    else None
                ),
            )
            audio = sampler.generate(gen_config, device=self._accelerator.device)
            output_path = output_dir / f"step_{step_value:06d}_{prompt_idx + 1:02d}.wav"
            self._save_validation_audio(audio, output_path)
            output_paths.append(output_path)

        sampling_ctx.cleanup()
        self._accelerator.wait_for_everyone()
        return output_paths

    def _run_distributed_validation(self, progress: TrainingProgress) -> list[Path]:
        sampled = self._sample_validation_audios(progress)

        if getattr(self._accelerator, "num_processes", 1) > 1:
            sampled = sorted(gather_object(sampled), key=lambda item: item[0])

        paths = [path for _, path in sampled]
        if getattr(self._accelerator, "is_main_process", True) and paths:
            self._log_validation_samples(paths, self._config.validation.prompts)
        if hasattr(self._accelerator, "wait_for_everyone"):
            self._accelerator.wait_for_everyone()
        return paths

    def _sample_validation_audios(self, progress: TrainingProgress) -> list[tuple[int, Path]]:
        if not self._config.validation.prompts:
            return []

        self._ensure_validation_components()
        prompts = self._config.validation.prompts
        rank = getattr(self._accelerator, "process_index", 0)
        world_size = getattr(self._accelerator, "num_processes", 1)
        rank_indices = list(range(rank, len(prompts), world_size))
        sampling_ctx = progress.start_sampling(
            num_prompts=len(rank_indices),
            num_steps=self._config.validation.inference_steps,
        )
        transformer = (
            self._accelerator.unwrap_model(self._transformer)
            if hasattr(self._accelerator, "unwrap_model")
            else self._transformer
        )
        sampler = ValidationSampler(
            transformer=transformer,
            audio_decoder=self._audio_decoder,
            vocoder=self._vocoder,
            sampling_context=sampling_ctx,
            scheduler=self._scheduler,
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: list[tuple[int, Path]] = []
        step_value = max(self._global_step, 0)

        for local_index, prompt_idx in enumerate(rank_indices):
            prompt = prompts[prompt_idx]
            sampling_ctx.start_video(local_index)
            gen_config = GenerationConfig(
                prompt=prompt,
                negative_prompt=self._config.validation.negative_prompt,
                audio_duration_seconds=self._config.validation.audio_duration_seconds,
                num_inference_steps=self._config.validation.inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                seed=self._config.validation.seed,
                cached_embeddings=(
                    self._cached_validation_embeddings[prompt_idx]
                    if self._cached_validation_embeddings is not None
                    else None
                ),
            )
            audio = sampler.generate(gen_config, device=self._accelerator.device)
            output_path = output_dir / f"step_{step_value:06d}_{prompt_idx + 1:02d}.wav"
            self._save_validation_audio(audio, output_path)
            output_paths.append((prompt_idx, output_path))

        sampling_ctx.cleanup()
        return output_paths

    @staticmethod
    def _save_validation_audio(audio: "Audio", output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        waveform = audio.waveform.detach().cpu()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(str(output_path), waveform, audio.sampling_rate)

    def _save_checkpoint(self) -> Path:
        checkpoint_dir = Path(self._config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        step_value = max(self._global_step, 0)
        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = getattr(self._accelerator, "distributed_type", None) == DistributedType.FSDP
        prefix = "lora" if is_lora else "model"
        checkpoint_path = checkpoint_dir / f"{prefix}_weights_step_{step_value:05d}.safetensors"

        if hasattr(self._accelerator, "wait_for_everyone"):
            self._accelerator.wait_for_everyone()
        full_state_dict = (
            self._accelerator.get_state_dict(self._transformer)
            if hasattr(self._accelerator, "get_state_dict")
            else None
        )

        if hasattr(self._accelerator, "unwrap_model"):
            try:
                unwrapped_model = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            except TypeError:
                unwrapped_model = self._accelerator.unwrap_model(self._transformer)
        else:
            unwrapped_model = self._transformer
        state_dict, metadata = self._build_checkpoint_payload(
            unwrapped_model,
            full_state_dict=full_state_dict if is_fsdp else None,
        )
        save_file(state_dict, checkpoint_path, metadata=metadata)
        self._checkpoint_paths.append(checkpoint_path)

        self._prune_old_checkpoints()
        self._save_training_state(checkpoint_dir)
        return checkpoint_path

    def _build_checkpoint_payload(
        self,
        model: torch.nn.Module,
        full_state_dict: dict[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, str] | None]:
        save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32

        if self._config.model.training_mode == "lora" and hasattr(model, "peft_config"):
            from peft import get_peft_model_state_dict  # noqa: PLC0415

            base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
            try:
                state_dict = get_peft_model_state_dict(base_model, state_dict=full_state_dict)
            except TypeError:
                state_dict = get_peft_model_state_dict(base_model)
            state_dict = {key.replace("base_model.model.", "", 1): value for key, value in state_dict.items()}
            state_dict = {f"diffusion_model.{key}": value for key, value in state_dict.items()}
            state_dict = {
                key: value.to(save_dtype) if isinstance(value, Tensor) else value for key, value in state_dict.items()
            }
            metadata = self._build_checkpoint_metadata()
            return state_dict, metadata

        state_dict = full_state_dict if full_state_dict is not None else model.state_dict()
        state_dict = {
            key: value.to(save_dtype) if isinstance(value, Tensor) else value for key, value in state_dict.items()
        }
        return state_dict, None

    def _build_checkpoint_metadata(self) -> dict[str, str]:
        raw_metadata = {}
        if hasattr(self._training_strategy, "get_checkpoint_metadata"):
            raw_metadata = self._training_strategy.get_checkpoint_metadata()
        return {key: str(value) for key, value in raw_metadata.items()}

    @staticmethod
    # ANN401 is acceptable here because this helper serializes heterogeneous config trees.
    # ruff: noqa: ANN401
    def _serialize_config_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: AudioTrainer._serialize_config_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [AudioTrainer._serialize_config_value(item) for item in value]
        if hasattr(value, "model_dump"):
            return AudioTrainer._serialize_config_value(value.model_dump())
        if hasattr(value, "__dict__") and not isinstance(value, (str, bytes, int, float, bool, type(None))):
            return {
                key: AudioTrainer._serialize_config_value(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        return value

    def _config_to_dict(self) -> dict[str, Any]:
        if hasattr(self._config, "model_dump"):
            return self._serialize_config_value(self._config.model_dump())
        return self._serialize_config_value(vars(self._config))

    def _save_config(self) -> None:
        config_path = Path(self._config.output_dir) / "training_config.yaml"
        model_dump = self._config_to_dict()
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(model_dump, file, sort_keys=False)

    def _capture_training_state(self) -> TrainingState:
        mode = getattr(self._config.checkpoints, "save_training_state", "minimal")
        torch_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        lora_config = getattr(self._config, "lora", None)
        return TrainingState(
            global_step=max(self._global_step, 0),
            config_fingerprint=ConfigFingerprint(
                optimizer_type=self._config.optimization.optimizer_type,
                scheduler_type=self._config.optimization.scheduler_type,
                training_mode=self._config.model.training_mode,
                lora_rank=lora_config.rank if lora_config is not None else None,
            ),
            rng_states=RngStates(torch_state=torch_state, cuda_state=cuda_state),
            lr_scheduler_state_dict=self._lr_scheduler.state_dict() if self._lr_scheduler is not None else None,
            optimizer_state_dict=(
                self._optimizer.state_dict()
                if mode == "full" and self._optimizer is not None
                else None
            ),
            wandb_run_id=self._wandb_run.id if self._wandb_run is not None else None,
        )

    def _save_training_state(self, checkpoint_dir: Path) -> None:
        mode = getattr(self._config.checkpoints, "save_training_state", "minimal")
        if mode == "off":
            return

        step_value = max(self._global_step, 0)
        state_path = checkpoint_dir / f"training_state_step_{step_value:05d}.pt"
        tmp_path = state_path.with_suffix(".pt.tmp")
        try:
            torch.save(self._capture_training_state().to_save_dict(), tmp_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        tmp_path.rename(state_path)

        if not self._training_state_paths or self._training_state_paths[-1] != state_path:
            self._training_state_paths.append(state_path)
        self._cleanup_training_states()

    def _cleanup_training_states(self) -> None:
        keep_last_n = self._config.checkpoints.keep_last_n
        if keep_last_n < 0:
            return
        if len(self._training_state_paths) <= keep_last_n:
            return

        to_remove = self._training_state_paths[:-keep_last_n]
        self._training_state_paths = self._training_state_paths[-keep_last_n:]
        for state_path in to_remove:
            state_path.unlink(missing_ok=True)

    def _load_checkpoint(self) -> None:
        load_path = getattr(self._config.model, "load_checkpoint", None)
        if not load_path:
            self._resume_state = (0, None)
            return

        checkpoint_path = self._find_checkpoint(load_path)
        if checkpoint_path is None:
            logger.warning(f"Could not find checkpoint at {load_path}")
            self._resume_state = (0, None)
            return

        self._loaded_checkpoint_path = checkpoint_path
        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        else:
            self._load_lora_checkpoint(checkpoint_path)
        self._resume_state = self._resolve_resume_state()

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        state_dict = load_file(checkpoint_path)
        self._transformer.load_state_dict(state_dict, strict=True)

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        state_dict = load_file(checkpoint_path)
        state_dict = {key.replace("diffusion_model.", "", 1): value for key, value in state_dict.items()}
        self._transformer.load_state_dict(state_dict, strict=False)

    def _resolve_resume_state(self) -> tuple[int, TrainingState | None]:
        if self._config.checkpoints.no_resume or self._loaded_checkpoint_path is None:
            return 0, None

        state = self._load_training_state(self._loaded_checkpoint_path)
        if state is None:
            return 0, None

        fingerprint = state.config_fingerprint
        config = self._config
        mismatches: list[str] = []
        if fingerprint.optimizer_type != config.optimization.optimizer_type:
            mismatches.append(f"optimizer_type: {fingerprint.optimizer_type} -> {config.optimization.optimizer_type}")
        if fingerprint.scheduler_type != config.optimization.scheduler_type:
            mismatches.append(f"scheduler_type: {fingerprint.scheduler_type} -> {config.optimization.scheduler_type}")
        if fingerprint.training_mode != config.model.training_mode:
            mismatches.append(f"training_mode: {fingerprint.training_mode} -> {config.model.training_mode}")
        if (
            config.model.training_mode == "lora"
            and config.lora is not None
            and fingerprint.lora_rank is not None
            and fingerprint.lora_rank != config.lora.rank
        ):
            mismatches.append(f"lora_rank: {fingerprint.lora_rank} -> {config.lora.rank}")
        if mismatches or state.global_step < 0:
            return 0, None
        return state.global_step, state

    def _restore_training_state(self, training_state: TrainingState) -> bool:
        try:
            if training_state.optimizer_state_dict is not None and self._optimizer is not None:
                self._optimizer.load_state_dict(training_state.optimizer_state_dict)
            if training_state.lr_scheduler_state_dict is not None and self._lr_scheduler is not None:
                self._lr_scheduler.load_state_dict(training_state.lr_scheduler_state_dict)
        except Exception:
            return False

        if getattr(self._accelerator, "num_processes", 1) == 1:
            torch.random.set_rng_state(training_state.rng_states.torch_state)
            if training_state.rng_states.cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(training_state.rng_states.cuda_state)
        return True

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        path = Path(checkpoint_path)
        if path.is_file():
            return path
        if path.is_dir():
            checkpoints = sorted(path.glob("*_weights_step_*.safetensors"))
            if not checkpoints:
                checkpoints = sorted(path.glob("checkpoint_step_*.pt"))
            if not checkpoints:
                return None
            return max(checkpoints, key=AudioTrainer._extract_step)
        return None

    @staticmethod
    def _extract_step(path: Path) -> int:
        match = re.search(r"step_(\d+)", path.name)
        return int(match.group(1)) if match else -1

    @staticmethod
    def _load_training_state(checkpoint_path: Path) -> TrainingState | None:
        step = AudioTrainer._extract_step(checkpoint_path)
        state_path = checkpoint_path.parent / f"training_state_step_{step:05d}.pt"
        if not state_path.exists():
            return None
        raw = torch.load(state_path, map_location="cpu", weights_only=False)
        return TrainingState.from_save_dict(raw)

    def _prune_old_checkpoints(self) -> None:
        keep_last_n = self._config.checkpoints.keep_last_n
        if keep_last_n < 0:
            return
        if len(self._checkpoint_paths) <= keep_last_n:
            return

        to_remove = self._checkpoint_paths[:-keep_last_n]
        self._checkpoint_paths = self._checkpoint_paths[-keep_last_n:]
        for checkpoint_path in to_remove:
            checkpoint_path.unlink(missing_ok=True)

    def _init_wandb(self, resume_run_id: str | None = None) -> None:
        if not getattr(self._config.wandb, "enabled", False):
            self._wandb_run = None
            return

        init_kwargs: dict[str, Any] = {
            "project": self._config.wandb.project,
            "entity": self._config.wandb.entity,
            "name": Path(self._config.output_dir).name,
            "tags": self._config.wandb.tags,
            "config": self._config_to_dict(),
        }
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "allow"
        self._wandb_run = wandb.init(**init_kwargs)

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    def _log_validation_samples(self, sample_paths: list[Path], prompts: list[str]) -> None:
        if not getattr(self._config.wandb, "log_validation_audio", False) or self._wandb_run is None:
            return

        samples = [
            wandb.Audio(str(path), caption=prompt)
            for path, prompt in zip(sample_paths, prompts, strict=True)
        ]
        self._wandb_run.log({"validation_samples": samples}, step=max(self._global_step, 0))
