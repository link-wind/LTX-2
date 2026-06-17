# ruff: noqa: ANN202, E501, N805, PIE807, RUF012

from __future__ import annotations

import sys
import types
from collections.abc import Generator
from pathlib import Path

import pytest
import torch

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
CORE_SRC = Path(__file__).resolve().parents[2] / "ltx-audio-core" / "src"


@pytest.fixture
def trainer_module() -> Generator[types.ModuleType, None, None]:
    sys.path.insert(0, str(PACKAGE_SRC))
    sys.path.insert(0, str(CORE_SRC))
    try:
        from ltx_audio_trainer import trainer  # noqa: PLC0415

        yield trainer
    finally:
        sys.path.pop(0)
        sys.path.pop(0)


def test_training_step_output_carries_loss_and_sigma(trainer_module: types.ModuleType) -> None:
    output = trainer_module.TrainingStepOutput(
        loss=torch.tensor([1.0, 2.0]),
        sigma=torch.tensor([0.1, 0.9]),
    )

    assert output.loss.shape == (2,)
    assert output.sigma.shape == (2,)


def test_audio_trainer_setup_accelerator_enables_find_unused_parameters(
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    captured: dict[str, object] = {}

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=2,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir="/tmp/out",
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "DistributedDataParallelKwargs",
        lambda **kwargs: captured.setdefault("ddp_kwargs", kwargs) or types.SimpleNamespace(**kwargs),
    )

    class _FakeAccelerator:
        num_processes = 1
        distributed_type = types.SimpleNamespace(value="NO")

    def _fake_accelerator(**kwargs: object) -> _FakeAccelerator:
        captured["accelerator_kwargs"] = kwargs
        return _FakeAccelerator()

    monkeypatch.setattr(trainer_module, "Accelerator", _fake_accelerator)
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )

    trainer_module.AudioTrainer(config)

    assert captured["ddp_kwargs"] == {"find_unused_parameters": True}
    assert captured["accelerator_kwargs"]["gradient_accumulation_steps"] == 2
    assert captured["accelerator_kwargs"]["mixed_precision"] == "bf16"


def test_audio_trainer_prepare_model_for_training_casts_fsdp_lora_to_fp32_and_uses_base_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    gradient_flags: list[bool] = []
    to_calls: list[torch.dtype] = []
    prepared: list[object] = []

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="lora", model_path="m", text_encoder_path="t"),
        lora=types.SimpleNamespace(rank=8, alpha=8, dropout=0.0, target_modules=["to_k"]),
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=True,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module.AudioTrainer, "_setup_lora", lambda _self: None)

    trainer = trainer_module.AudioTrainer(config)

    class _BaseModel:
        def set_gradient_checkpointing(self, flag: bool) -> None:
            gradient_flags.append(flag)

    class _FakeTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = _BaseModel()

        def parameters(self):
            return [parameter]

        def get_base_model(self) -> _BaseModel:
            return self.base_model

        def to(self, dtype=None, **_kwargs: object):  # noqa: ANN001
            to_calls.append(dtype)
            return self

    trainer._transformer = _FakeTransformer()
    trainer._accelerator = types.SimpleNamespace(
        distributed_type="FSDP",
        prepare=lambda model: prepared.append(model) or model,
    )

    class _FakeDistributedType:
        FSDP = "FSDP"

    monkeypatch.setattr(trainer_module, "DistributedType", _FakeDistributedType)

    trainer._prepare_model_for_training()

    assert to_calls == [torch.float32]
    assert gradient_flags == [True]
    assert prepared == [trainer._transformer]


def test_audio_trainer_builds_optimizer_for_trainable_param_groups(
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir="/tmp/out",
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )

    trainer = trainer_module.AudioTrainer(config)
    trainer._trainable_params = [parameter]

    trainer._init_optimizer()

    assert trainer._optimizer is not None
    assert trainer._optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_audio_trainer_builds_precomputed_dataset_from_strategy_sources(
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    captured: dict[str, object] = {}
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=2,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=3),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir="/tmp/out",
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(
            get_data_sources=lambda: {
                "audio_latents": "latent_conditions",
                "conditions": "text_conditions",
            }
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "PrecomputedDataset",
        lambda data_root, data_sources: captured.update({"data_root": data_root, "data_sources": data_sources})
        or [1, 2, 3],
    )

    trainer = trainer_module.AudioTrainer(config)
    trainer._accelerator = types.SimpleNamespace(
        prepare=lambda dataloader: dataloader,
        num_processes=1,
    )

    trainer._init_dataloader()

    assert captured["data_root"] == "/tmp/pre"
    assert captured["data_sources"] == {
        "audio_latents": "latent_conditions",
        "conditions": "text_conditions",
    }
    assert trainer._dataset == [1, 2, 3]


def test_audio_trainer_run_validation_saves_audio_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    saved_calls: list[tuple[str, int]] = []
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=2,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(
            interval=1,
            skip_initial_validation=False,
            prompts=["rain"],
            negative_prompt="",
            audio_duration_seconds=0.25,
            seed=0,
            inference_steps=2,
            guidance_scale=3.0,
        ),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(
            get_data_sources=lambda: {
                "audio_latents": "latent_conditions",
                "conditions": "text_conditions",
            }
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler="scheduler",
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(
        trainer_module.AudioTrainer,
        "_cache_validation_embeddings",
        lambda _self: ["cached"],
    )
    monkeypatch.setattr(
        trainer_module.AudioTrainer,
        "_ensure_validation_components",
        lambda _self: None,
    )

    class _FakeSampler:
        def __init__(self, **_kwargs) -> None:
            pass

        def generate(self, config, device) -> types.SimpleNamespace:  # noqa: ANN001, ARG002
            return types.SimpleNamespace(
                waveform=torch.ones(1, 32),
                sampling_rate=16_000,
            )

    def _save_stub(output_path: str, _waveform: torch.Tensor, sample_rate: int) -> None:
        saved_calls.append((output_path, sample_rate))

    monkeypatch.setattr(trainer_module, "ValidationSampler", _FakeSampler)
    monkeypatch.setattr(
        trainer_module.torchaudio,
        "save",
        _save_stub,
    )

    trainer = trainer_module.AudioTrainer(config)
    trainer._audio_decoder = object()
    trainer._vocoder = object()
    trainer._accelerator = types.SimpleNamespace(
        is_local_main_process=True,
        wait_for_everyone=lambda: None,
        device=torch.device("cpu"),
        num_processes=1,
    )
    trainer._global_step = 3

    class _SamplingCtx:
        def start_video(self, _idx: int) -> None:
            return None

        def advance_step(self) -> None:
            return None

        def cleanup(self) -> None:
            return None

    def _start_sampling(*, num_prompts: int, num_steps: int) -> _SamplingCtx:  # noqa: ARG001
        return _SamplingCtx()

    progress = types.SimpleNamespace(start_sampling=_start_sampling)

    paths = trainer._run_validation(progress)

    assert len(paths) == 1
    assert paths[0].name == "step_000003_01.wav"
    assert saved_calls == [(str(paths[0]), 16_000)]


def test_audio_trainer_save_checkpoint_uses_safetensors_for_full_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    saved: dict[str, object] = {}

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(
            interval=None,
            keep_last_n=1,
            precision="bfloat16",
            no_resume=True,
            save_training_state="minimal",
        ),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "save_file",
        lambda state_dict, path, metadata=None: saved.update(
            {"state_dict": state_dict, "path": Path(path), "metadata": metadata}
        ),
    )
    monkeypatch.setattr(
        trainer_module.torch,
        "save",
        lambda _payload, path, **_kwargs: Path(path).write_bytes(b"state"),
    )

    trainer = trainer_module.AudioTrainer(config)
    trainer._global_step = 7
    trainer._optimizer = types.SimpleNamespace(state_dict=lambda: {"optimizer": 1})
    trainer._lr_scheduler = types.SimpleNamespace(state_dict=lambda: {"scheduler": 1})
    trainer._accelerator = types.SimpleNamespace(
        unwrap_model=lambda model: model,
    )
    trainer._transformer = types.SimpleNamespace(
        state_dict=lambda: {"weight": torch.ones(2, dtype=torch.float32)},
    )

    checkpoint_path = trainer._save_checkpoint()

    assert checkpoint_path.name == "model_weights_step_00007.safetensors"
    assert saved["path"] == checkpoint_path
    assert saved["metadata"] is None
    assert saved["state_dict"]["weight"].dtype == torch.bfloat16


def test_audio_trainer_save_checkpoint_formats_lora_weights_for_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    saved: dict[str, object] = {}

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="lora", model_path="m", text_encoder_path="t"),
        lora=types.SimpleNamespace(rank=8, alpha=8, dropout=0.0, target_modules=["to_k"]),
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(
            interval=None,
            keep_last_n=1,
            precision="bfloat16",
            no_resume=True,
            save_training_state="minimal",
        ),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(
            get_data_sources=lambda: {"audio_latents": "latent_conditions"},
            get_checkpoint_metadata=lambda: {"modality": "audio"},
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "save_file",
        lambda state_dict, path, metadata=None: saved.update(
            {"state_dict": state_dict, "path": Path(path), "metadata": metadata}
        ),
    )
    monkeypatch.setattr(
        trainer_module.torch,
        "save",
        lambda _payload, path, **_kwargs: Path(path).write_bytes(b"state"),
    )

    peft_module = types.ModuleType("peft")
    peft_module.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    peft_module.get_peft_model = lambda model, _config: model
    def _get_peft_model_state_dict(_model: object, _state_dict: object = None) -> dict[str, torch.Tensor]:
        return {
            "base_model.model.transformer_blocks.0.attn.to_k.lora_A.weight": torch.ones(2, 2, dtype=torch.float32),
            "base_model.model.transformer_blocks.0.attn.to_k.lora_B.weight": torch.ones(2, 2, dtype=torch.float32),
        }

    peft_module.get_peft_model_state_dict = _get_peft_model_state_dict
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    trainer = trainer_module.AudioTrainer(config)
    trainer._global_step = 11
    trainer._optimizer = types.SimpleNamespace(state_dict=lambda: {"optimizer": 1})
    trainer._lr_scheduler = types.SimpleNamespace(state_dict=lambda: {"scheduler": 1})
    trainer._accelerator = types.SimpleNamespace(
        unwrap_model=lambda model: model,
    )
    trainer._transformer = types.SimpleNamespace(peft_config={"default": object()})

    checkpoint_path = trainer._save_checkpoint()

    assert checkpoint_path.name == "lora_weights_step_00011.safetensors"
    assert saved["path"] == checkpoint_path
    assert saved["metadata"] == {"modality": "audio"}
    assert set(saved["state_dict"]) == {
        "diffusion_model.transformer_blocks.0.attn.to_k.lora_A.weight",
        "diffusion_model.transformer_blocks.0.attn.to_k.lora_B.weight",
    }
    assert all(value.dtype == torch.bfloat16 for value in saved["state_dict"].values())


def test_audio_trainer_save_checkpoint_minimal_training_state_omits_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    saved_payloads: list[dict[str, object]] = []

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(
            interval=None,
            keep_last_n=2,
            precision="bfloat16",
            no_resume=True,
            save_training_state="minimal",
        ),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module, "save_file", lambda *_args, **_kwargs: None)

    def _fake_torch_save(payload: dict[str, object], path: Path, **_kwargs: object) -> None:
        saved_payloads.append({"payload": payload, "path": Path(path)})
        Path(path).write_bytes(b"state")

    monkeypatch.setattr(trainer_module.torch, "save", _fake_torch_save)

    trainer = trainer_module.AudioTrainer(config)
    trainer._global_step = 7
    trainer._optimizer = types.SimpleNamespace(state_dict=lambda: {"optimizer": 1})
    trainer._lr_scheduler = types.SimpleNamespace(state_dict=lambda: {"scheduler": 1})
    trainer._accelerator = types.SimpleNamespace(
        unwrap_model=lambda model: model,
    )
    trainer._transformer = types.SimpleNamespace(
        state_dict=lambda: {"weight": torch.ones(2, dtype=torch.float32)},
    )

    trainer._save_checkpoint()

    assert len(saved_payloads) == 1
    payload = saved_payloads[0]["payload"]
    assert payload["global_step"] == 7
    assert payload["lr_scheduler_state_dict"] == {"scheduler": 1}
    assert "optimizer_state_dict" not in payload
    assert (tmp_path / "checkpoints" / "training_state_step_00007.pt").exists()


def test_audio_trainer_save_checkpoint_full_training_state_includes_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    saved_payloads: list[dict[str, object]] = []

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(
            interval=None,
            keep_last_n=2,
            precision="bfloat16",
            no_resume=True,
            save_training_state="full",
        ),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module, "save_file", lambda *_args, **_kwargs: None)

    def _fake_torch_save(payload: dict[str, object], path: Path, **_kwargs: object) -> None:
        saved_payloads.append({"payload": payload, "path": Path(path)})
        Path(path).write_bytes(b"state")

    monkeypatch.setattr(trainer_module.torch, "save", _fake_torch_save)

    trainer = trainer_module.AudioTrainer(config)
    trainer._global_step = 8
    trainer._optimizer = types.SimpleNamespace(state_dict=lambda: {"optimizer": 9})
    trainer._lr_scheduler = types.SimpleNamespace(state_dict=lambda: {"scheduler": 2})
    trainer._accelerator = types.SimpleNamespace(
        unwrap_model=lambda model: model,
    )
    trainer._transformer = types.SimpleNamespace(
        state_dict=lambda: {"weight": torch.ones(2, dtype=torch.float32)},
    )

    trainer._save_checkpoint()

    assert len(saved_payloads) == 1
    payload = saved_payloads[0]["payload"]
    assert payload["optimizer_state_dict"] == {"optimizer": 9}
    assert payload["lr_scheduler_state_dict"] == {"scheduler": 2}


def test_audio_trainer_save_checkpoint_prunes_old_training_state_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(
            interval=None,
            keep_last_n=1,
            precision="bfloat16",
            no_resume=True,
            save_training_state="minimal",
        ),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )

    def _fake_save_file(_state_dict: dict[str, object], path: Path, metadata=None) -> None:  # noqa: ANN001, ARG001
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"weights")

    def _fake_torch_save(payload: dict[str, object], path: Path, **_kwargs: object) -> None:
        Path(path).write_bytes(str(payload["global_step"]).encode("utf-8"))

    monkeypatch.setattr(trainer_module, "save_file", _fake_save_file)
    monkeypatch.setattr(trainer_module.torch, "save", _fake_torch_save)

    trainer = trainer_module.AudioTrainer(config)
    trainer._optimizer = types.SimpleNamespace(state_dict=lambda: {"optimizer": 1})
    trainer._lr_scheduler = types.SimpleNamespace(state_dict=lambda: {"scheduler": 1})
    trainer._accelerator = types.SimpleNamespace(
        unwrap_model=lambda model: model,
    )
    trainer._transformer = types.SimpleNamespace(
        state_dict=lambda: {"weight": torch.ones(2, dtype=torch.float32)},
    )

    trainer._global_step = 1
    trainer._save_checkpoint()
    trainer._global_step = 2
    trainer._save_checkpoint()

    checkpoints_dir = tmp_path / "checkpoints"
    assert not (checkpoints_dir / "training_state_step_00001.pt").exists()
    assert (checkpoints_dir / "training_state_step_00002.pt").exists()


def test_audio_trainer_save_checkpoint_uses_accelerator_state_dict_for_fsdp_lora(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    captured: dict[str, object] = {}

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="lora", model_path="m", text_encoder_path="t"),
        lora=types.SimpleNamespace(rank=8, alpha=8, dropout=0.0, target_modules=["to_k"]),
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(
            interval=None,
            keep_last_n=1,
            precision="bfloat16",
            no_resume=True,
            save_training_state="off",
        ),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(
            get_data_sources=lambda: {"audio_latents": "latent_conditions"},
            get_checkpoint_metadata=lambda: {"modality": "audio"},
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "save_file",
        lambda state_dict, path, metadata=None: captured.update(
            {"saved_state_dict": state_dict, "saved_path": Path(path), "saved_metadata": metadata}
        ),
    )

    peft_module = types.ModuleType("peft")
    peft_module.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    peft_module.get_peft_model = lambda model, _config: model

    def _get_peft_model_state_dict(_model: object, state_dict: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        captured["peft_input_state_dict"] = state_dict
        return {
            "base_model.model.transformer_blocks.0.attn.to_k.lora_A.weight": torch.ones(2, 2, dtype=torch.float32),
        }

    peft_module.get_peft_model_state_dict = _get_peft_model_state_dict
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    trainer = trainer_module.AudioTrainer(config)
    trainer._global_step = 5
    trainer._optimizer = types.SimpleNamespace(state_dict=lambda: {"optimizer": 1})
    trainer._lr_scheduler = None

    class _FakePeftModel:
        peft_config = {"default": object()}

    trainer._transformer = _FakePeftModel()

    class _FakeDistributedType:
        FSDP = "FSDP"

    monkeypatch.setattr(trainer_module, "DistributedType", _FakeDistributedType)
    trainer._accelerator = types.SimpleNamespace(
        distributed_type="FSDP",
        wait_for_everyone=lambda: captured.setdefault("waited", True),
        get_state_dict=lambda _model: {"wrapped.weight": torch.full((1,), 3.0)},
        unwrap_model=lambda model, keep_torch_compile=False: captured.update(
            {"keep_torch_compile": keep_torch_compile}
        )
        or model,
    )

    checkpoint_path = trainer._save_checkpoint()

    assert checkpoint_path.name == "lora_weights_step_00005.safetensors"
    assert captured["waited"] is True
    assert captured["keep_torch_compile"] is False
    assert captured["peft_input_state_dict"] == {"wrapped.weight": torch.full((1,), 3.0)}
    assert captured["saved_metadata"] == {"modality": "audio"}


def test_audio_trainer_find_checkpoint_prefers_latest_safetensors(
    tmp_path: Path,
    trainer_module: types.ModuleType,
) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    older = checkpoints_dir / "model_weights_step_00003.safetensors"
    newer = checkpoints_dir / "model_weights_step_00009.safetensors"
    older.touch()
    newer.touch()

    found = trainer_module.AudioTrainer._find_checkpoint(checkpoints_dir)

    assert found == newer


def test_audio_trainer_load_checkpoint_resolves_resume_state_with_matching_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    checkpoint_path = tmp_path / "checkpoints" / "model_weights_step_00009.safetensors"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"weights")
    loaded: list[tuple[dict[str, torch.Tensor], bool]] = []

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(
            training_mode="full",
            model_path="m",
            text_encoder_path="t",
            load_checkpoint=checkpoint_path,
        ),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=12,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(
            interval=None,
            keep_last_n=1,
            precision="bfloat16",
            no_resume=False,
            save_training_state="minimal",
        ),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path / "out"),
    )

    training_state = trainer_module.TrainingState(
        global_step=9,
        config_fingerprint=trainer_module.ConfigFingerprint(
            optimizer_type="adamw",
            scheduler_type="constant",
            training_mode="full",
            lora_rank=None,
        ),
        rng_states=trainer_module.RngStates(torch_state=torch.tensor([1], dtype=torch.uint8), cuda_state=None),
        lr_scheduler_state_dict={"scheduler": 1},
        optimizer_state_dict={"optimizer": 2},
        wandb_run_id="run-1",
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(
                parameters=lambda: [parameter],
                load_state_dict=lambda state_dict, strict=True: loaded.append((state_dict, strict)),
            ),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module, "load_file", lambda _path: {"weight": torch.ones(1)})
    monkeypatch.setattr(trainer_module.AudioTrainer, "_load_training_state", staticmethod(lambda _path: training_state))

    trainer = trainer_module.AudioTrainer(config)

    assert loaded == [({"weight": torch.ones(1)}, True)]
    assert trainer._loaded_checkpoint_path == checkpoint_path
    assert trainer._resume_state == (9, training_state)


def test_audio_trainer_load_checkpoint_drops_resume_state_on_fingerprint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    checkpoint_path = tmp_path / "checkpoints" / "model_weights_step_00009.safetensors"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"weights")

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(
            training_mode="full",
            model_path="m",
            text_encoder_path="t",
            load_checkpoint=checkpoint_path,
        ),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=12,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(
            interval=None,
            keep_last_n=1,
            precision="bfloat16",
            no_resume=False,
            save_training_state="minimal",
        ),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path / "out"),
    )

    training_state = trainer_module.TrainingState(
        global_step=9,
        config_fingerprint=trainer_module.ConfigFingerprint(
            optimizer_type="adamw8bit",
            scheduler_type="constant",
            training_mode="full",
            lora_rank=None,
        ),
        rng_states=trainer_module.RngStates(torch_state=torch.tensor([1], dtype=torch.uint8), cuda_state=None),
        lr_scheduler_state_dict={"scheduler": 1},
        optimizer_state_dict={"optimizer": 2},
        wandb_run_id="run-1",
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(
                parameters=lambda: [parameter],
                load_state_dict=lambda *_args, **_kwargs: None,
            ),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module, "load_file", lambda _path: {"weight": torch.ones(1)})
    monkeypatch.setattr(trainer_module.AudioTrainer, "_load_training_state", staticmethod(lambda _path: training_state))

    trainer = trainer_module.AudioTrainer(config)

    assert trainer._resume_state == (0, None)


def test_audio_trainer_restore_training_state_returns_false_on_optimizer_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path / "out"),
    )

    training_state = trainer_module.TrainingState(
        global_step=3,
        config_fingerprint=trainer_module.ConfigFingerprint(
            optimizer_type="adamw",
            scheduler_type="constant",
            training_mode="full",
            lora_rank=None,
        ),
        rng_states=trainer_module.RngStates(torch_state=torch.tensor([1], dtype=torch.uint8), cuda_state=None),
        lr_scheduler_state_dict={"scheduler": 1},
        optimizer_state_dict={"optimizer": 2},
        wandb_run_id="run-1",
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )

    trainer = trainer_module.AudioTrainer(config)
    trainer._optimizer = types.SimpleNamespace(load_state_dict=lambda _state: (_ for _ in ()).throw(RuntimeError("boom")))
    trainer._lr_scheduler = types.SimpleNamespace(load_state_dict=lambda _state: None)
    trainer._accelerator = types.SimpleNamespace(num_processes=1)

    restored = trainer._restore_training_state(training_state)

    assert restored is False


def test_audio_trainer_train_initializes_wandb_logs_and_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    metrics_log: list[dict[str, float]] = []
    wandb_log: list[object] = []

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t", load_checkpoint=None),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(
            interval=None,
            keep_last_n=1,
            precision="bfloat16",
            no_resume=True,
            save_training_state="off",
        ),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(
            enabled=True,
            project="audio-trainer",
            entity="team",
            tags=["audio"],
            log_validation_audio=True,
        ),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    class _FakeRun:
        id = "wandb-run-1"

        def log(self, payload: object, step: int | None = None) -> None:
            wandb_log.append((payload, step))

        def finish(self) -> None:
            wandb_log.append("finished")

    class _FakeAccelerator:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.num_processes = 1
            self.is_local_main_process = True
            self.sync_gradients = True

        def prepare(self, *items: object) -> object:
            if len(items) == 1:
                return items[0]
            return items

        def accumulate(self, _model: object):
            class _Ctx:
                def __enter__(self_inner) -> None:
                    return None

                def __exit__(self_inner, *args: object) -> None:
                    return None

            return _Ctx()

        def backward(self, _loss: torch.Tensor) -> None:
            return None

        def clip_grad_norm_(self, _params: object, _max_norm: float) -> None:
            return None

        def unwrap_model(self, model: object) -> object:
            return model

        def wait_for_everyone(self) -> None:
            return None

        def end_training(self) -> None:
            return None

    class _FakeOptimizer:
        param_groups = [{"lr": 1e-4}]

        def step(self) -> None:
            return None

        def zero_grad(self) -> None:
            return None

        def state_dict(self) -> dict[str, int]:
            return {"optimizer": 1}

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = parameter

        def forward(self, _audio: object) -> torch.Tensor:
            return torch.zeros(1, 2, 3)

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(
            get_data_sources=lambda: {"audio_latents": "latent_conditions"},
            prepare_training_inputs=lambda _batch, _sampler: types.SimpleNamespace(
                audio=types.SimpleNamespace(sigma=torch.tensor([0.5]))
            ),
            compute_loss=lambda _pred, _inputs: torch.tensor([1.0], requires_grad=True),
            get_checkpoint_metadata=lambda: {},
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=_FakeModel(),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module.AudioTrainer, "_setup_accelerator", lambda self: setattr(self, "_accelerator", _FakeAccelerator()))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_prepare_model_for_training", lambda self: setattr(self, "_model_prepared", True))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_init_optimizer", lambda self: setattr(self, "_optimizer", _FakeOptimizer()) or setattr(self, "_lr_scheduler", None))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_init_dataloader", lambda self: setattr(self, "_dataloader", [{"dummy": torch.tensor([1.0])}]))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_init_timestep_sampler", lambda self: setattr(self, "_timestep_sampler", object()))
    def _save_checkpoint_stub(_trainer: object) -> Path:
        return tmp_path / "checkpoints" / "model_weights_step_00001.safetensors"

    def _log_metrics_stub(_trainer: object, metrics: dict[str, float]) -> None:
        metrics_log.append(metrics)

    monkeypatch.setattr(trainer_module.AudioTrainer, "_save_checkpoint", _save_checkpoint_stub)
    monkeypatch.setattr(trainer_module.AudioTrainer, "_log_metrics", _log_metrics_stub)

    wandb_module = types.SimpleNamespace(init=lambda **kwargs: wandb_log.append(kwargs) or _FakeRun(), Audio=lambda path, caption=None: (path, caption))
    monkeypatch.setattr(trainer_module, "wandb", wandb_module)

    trainer = trainer_module.AudioTrainer(config)
    trainer.train(disable_progress_bars=True)

    assert wandb_log[0]["project"] == "audio-trainer"
    assert any("train/loss" in metrics for metrics in metrics_log)
    assert any("stats/steps_per_second" in metrics for metrics in metrics_log)
    assert wandb_log[-1] == "finished"


def test_audio_trainer_train_pushes_to_hub_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    pushed: list[tuple[Path, list[Path] | None]] = []

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t", load_checkpoint=None),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(
            interval=None,
            keep_last_n=1,
            precision="bfloat16",
            no_resume=True,
            save_training_state="off",
        ),
        hub=types.SimpleNamespace(push_to_hub=True, hub_model_id="user/audio-model"),
        wandb=types.SimpleNamespace(enabled=False, project="audio-trainer", entity=None, tags=[], log_validation_audio=True),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    class _FakeAccelerator:
        device = torch.device("cpu")
        num_processes = 1
        is_local_main_process = True
        sync_gradients = True

        def prepare(self, *items: object) -> object:
            if len(items) == 1:
                return items[0]
            return items

        def accumulate(self, _model: object):
            class _Ctx:
                def __enter__(self_inner) -> None:
                    return None

                def __exit__(self_inner, *args: object) -> None:
                    return None

            return _Ctx()

        def backward(self, _loss: torch.Tensor) -> None:
            return None

        def clip_grad_norm_(self, _params: object, _max_norm: float) -> None:
            return None

        def unwrap_model(self, model: object) -> object:
            return model

        def wait_for_everyone(self) -> None:
            return None

        def end_training(self) -> None:
            return None

    class _FakeOptimizer:
        param_groups = [{"lr": 1e-4}]

        def step(self) -> None:
            return None

        def zero_grad(self) -> None:
            return None

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = parameter

        def forward(self, _audio: object) -> torch.Tensor:
            return torch.zeros(1, 2, 3)

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(
            get_data_sources=lambda: {"audio_latents": "latent_conditions"},
            prepare_training_inputs=lambda _batch, _sampler: types.SimpleNamespace(
                audio=types.SimpleNamespace(sigma=torch.tensor([0.5]))
            ),
            compute_loss=lambda _pred, _inputs: torch.tensor([1.0], requires_grad=True),
            get_checkpoint_metadata=lambda: {},
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=_FakeModel(),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module.AudioTrainer, "_setup_accelerator", lambda self: setattr(self, "_accelerator", _FakeAccelerator()))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_prepare_model_for_training", lambda self: setattr(self, "_model_prepared", True))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_init_optimizer", lambda self: setattr(self, "_optimizer", _FakeOptimizer()) or setattr(self, "_lr_scheduler", None))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_init_dataloader", lambda self: setattr(self, "_dataloader", [{"dummy": torch.tensor([1.0])}]))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_init_timestep_sampler", lambda self: setattr(self, "_timestep_sampler", object()))
    def _save_checkpoint_stub(_trainer: object) -> Path:
        return tmp_path / "checkpoints" / "model_weights_step_00001.safetensors"

    def _push_to_hub_stub(weights_path: Path, sample_paths: list[Path] | None, _cfg: object) -> None:
        pushed.append((weights_path, sample_paths))

    monkeypatch.setattr(trainer_module.AudioTrainer, "_save_checkpoint", _save_checkpoint_stub)
    monkeypatch.setattr(trainer_module, "push_to_hub", _push_to_hub_stub)

    trainer = trainer_module.AudioTrainer(config)
    trainer.train(disable_progress_bars=True)

    assert pushed == [(tmp_path / "checkpoints" / "model_weights_step_00001.safetensors", None)]


def test_audio_trainer_train_writes_training_config_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t", load_checkpoint=None),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(
            interval=None,
            keep_last_n=1,
            precision="bfloat16",
            no_resume=True,
            save_training_state="off",
        ),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False, project="audio-trainer", entity=None, tags=[], log_validation_audio=True),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
        model_dump=lambda: {"seed": 42, "output_dir": str(tmp_path)},
    )

    class _FakeAccelerator:
        device = torch.device("cpu")
        num_processes = 1
        is_local_main_process = True
        sync_gradients = True

        def prepare(self, *items: object) -> object:
            if len(items) == 1:
                return items[0]
            return items

        def accumulate(self, _model: object):
            class _Ctx:
                def __enter__(self_inner) -> None:
                    return None

                def __exit__(self_inner, *args: object) -> None:
                    return None

            return _Ctx()

        def backward(self, _loss: torch.Tensor) -> None:
            return None

        def clip_grad_norm_(self, _params: object, _max_norm: float) -> None:
            return None

        def unwrap_model(self, model: object) -> object:
            return model

        def wait_for_everyone(self) -> None:
            return None

        def end_training(self) -> None:
            return None

    class _FakeOptimizer:
        param_groups = [{"lr": 1e-4}]

        def step(self) -> None:
            return None

        def zero_grad(self) -> None:
            return None

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = parameter

        def forward(self, _audio: object) -> torch.Tensor:
            return torch.zeros(1, 2, 3)

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(
            get_data_sources=lambda: {"audio_latents": "latent_conditions"},
            prepare_training_inputs=lambda _batch, _sampler: types.SimpleNamespace(
                audio=types.SimpleNamespace(sigma=torch.tensor([0.5]))
            ),
            compute_loss=lambda _pred, _inputs: torch.tensor([1.0], requires_grad=True),
            get_checkpoint_metadata=lambda: {},
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=_FakeModel(),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module.AudioTrainer, "_setup_accelerator", lambda self: setattr(self, "_accelerator", _FakeAccelerator()))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_prepare_model_for_training", lambda self: setattr(self, "_model_prepared", True))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_init_optimizer", lambda self: setattr(self, "_optimizer", _FakeOptimizer()) or setattr(self, "_lr_scheduler", None))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_init_dataloader", lambda self: setattr(self, "_dataloader", [{"dummy": torch.tensor([1.0])}]))
    monkeypatch.setattr(trainer_module.AudioTrainer, "_init_timestep_sampler", lambda self: setattr(self, "_timestep_sampler", object()))
    def _save_checkpoint_stub(_trainer: object) -> Path:
        return tmp_path / "checkpoints" / "model_weights_step_00001.safetensors"

    monkeypatch.setattr(trainer_module.AudioTrainer, "_save_checkpoint", _save_checkpoint_stub)

    trainer = trainer_module.AudioTrainer(config)
    trainer.train(disable_progress_bars=True)

    config_path = tmp_path / "training_config.yaml"
    assert config_path.exists()
    assert "seed: 42" in config_path.read_text(encoding="utf-8")


def test_audio_trainer_run_distributed_validation_gathers_and_sorts_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    gathered_inputs: list[list[tuple[int, Path]]] = []

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(
            interval=1,
            skip_initial_validation=False,
            prompts=["p0", "p1", "p2"],
            negative_prompt="",
            audio_duration_seconds=0.25,
            seed=0,
            inference_steps=2,
            guidance_scale=3.0,
        ),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(
            get_data_sources=lambda: {
                "audio_latents": "latent_conditions",
                "conditions": "text_conditions",
            }
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler="scheduler",
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module.AudioTrainer, "_cache_validation_embeddings", lambda _self: ["c0", "c1", "c2"])

    def _sample_validation_audios(_self: object, _progress: object) -> list[tuple[int, Path]]:
        return [(2, tmp_path / "step_000003_03.wav"), (0, tmp_path / "step_000003_01.wav")]

    def _gather_object(payload: list[tuple[int, Path]]) -> list[tuple[int, Path]]:
        gathered_inputs.append(payload)
        return [*payload, (1, tmp_path / "step_000003_02.wav")]

    monkeypatch.setattr(trainer_module.AudioTrainer, "_sample_validation_audios", _sample_validation_audios)
    monkeypatch.setattr(trainer_module, "gather_object", _gather_object)

    trainer = trainer_module.AudioTrainer(config)
    trainer._accelerator = types.SimpleNamespace(
        num_processes=2,
        is_main_process=True,
        wait_for_everyone=lambda: None,
    )

    paths = trainer._run_distributed_validation(progress=object())

    assert gathered_inputs == [[(2, tmp_path / "step_000003_03.wav"), (0, tmp_path / "step_000003_01.wav")]]
    assert paths == [
        tmp_path / "step_000003_01.wav",
        tmp_path / "step_000003_02.wav",
        tmp_path / "step_000003_03.wav",
    ]


def test_audio_trainer_run_distributed_validation_logs_wandb_audio_on_main_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    logged_payloads: list[tuple[object, int | None]] = []

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(
            interval=1,
            skip_initial_validation=False,
            prompts=["p0", "p1", "p2"],
            negative_prompt="",
            audio_duration_seconds=0.25,
            seed=0,
            inference_steps=2,
            guidance_scale=3.0,
        ),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(
            enabled=True,
            project="audio-trainer",
            entity=None,
            tags=[],
            log_validation_audio=True,
        ),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(
            get_data_sources=lambda: {
                "audio_latents": "latent_conditions",
                "conditions": "text_conditions",
            }
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler="scheduler",
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module.AudioTrainer, "_cache_validation_embeddings", lambda _self: ["c0", "c1", "c2"])
    monkeypatch.setattr(
        trainer_module.AudioTrainer,
        "_sample_validation_audios",
        lambda _self, _progress: [
            (2, tmp_path / "step_000003_03.wav"),
            (0, tmp_path / "step_000003_01.wav"),
        ],
    )
    monkeypatch.setattr(
        trainer_module,
        "gather_object",
        lambda payload: [*payload, (1, tmp_path / "step_000003_02.wav")],
    )
    monkeypatch.setattr(
        trainer_module,
        "wandb",
        types.SimpleNamespace(Audio=lambda path, caption=None: ("audio", path, caption)),
    )

    trainer = trainer_module.AudioTrainer(config)
    trainer._global_step = 3
    trainer._wandb_run = types.SimpleNamespace(log=lambda payload, step=None: logged_payloads.append((payload, step)))
    trainer._accelerator = types.SimpleNamespace(
        num_processes=2,
        is_main_process=True,
        wait_for_everyone=lambda: None,
    )

    trainer._run_distributed_validation(progress=object())

    assert logged_payloads == [
        (
            {
                "validation_samples": [
                    ("audio", str(tmp_path / "step_000003_01.wav"), "p0"),
                    ("audio", str(tmp_path / "step_000003_02.wav"), "p1"),
                    ("audio", str(tmp_path / "step_000003_03.wav"), "p2"),
                ]
            },
            3,
        )
    ]


def test_audio_trainer_run_distributed_validation_skips_wandb_audio_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    logged_payloads: list[tuple[object, int | None]] = []

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(
            interval=1,
            skip_initial_validation=False,
            prompts=["p0"],
            negative_prompt="",
            audio_duration_seconds=0.25,
            seed=0,
            inference_steps=2,
            guidance_scale=3.0,
        ),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(
            enabled=True,
            project="audio-trainer",
            entity=None,
            tags=[],
            log_validation_audio=False,
        ),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(
            get_data_sources=lambda: {
                "audio_latents": "latent_conditions",
                "conditions": "text_conditions",
            }
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler="scheduler",
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module.AudioTrainer, "_cache_validation_embeddings", lambda _self: ["c0"])
    monkeypatch.setattr(
        trainer_module.AudioTrainer,
        "_sample_validation_audios",
        lambda _self, _progress: [(0, tmp_path / "step_000003_01.wav")],
    )
    monkeypatch.setattr(
        trainer_module,
        "wandb",
        types.SimpleNamespace(Audio=lambda path, caption=None: ("audio", path, caption)),
    )

    trainer = trainer_module.AudioTrainer(config)
    trainer._global_step = 3
    trainer._wandb_run = types.SimpleNamespace(log=lambda payload, step=None: logged_payloads.append((payload, step)))
    trainer._accelerator = types.SimpleNamespace(
        num_processes=1,
        is_main_process=True,
        wait_for_everyone=lambda: None,
    )

    trainer._run_distributed_validation(progress=object())

    assert logged_payloads == []


def test_audio_trainer_sample_validation_audios_uses_rank_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    generated_prompts: list[str] = []
    saved_calls: list[tuple[str, int]] = []

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=2,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(
            interval=1,
            skip_initial_validation=False,
            prompts=["p0", "p1", "p2", "p3"],
            negative_prompt="",
            audio_duration_seconds=0.25,
            seed=0,
            inference_steps=2,
            guidance_scale=3.0,
        ),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(
            get_data_sources=lambda: {
                "audio_latents": "latent_conditions",
                "conditions": "text_conditions",
            }
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler="scheduler",
            unload_text_encoder=lambda **_kw: None,
        ),
    )
    monkeypatch.setattr(trainer_module.AudioTrainer, "_cache_validation_embeddings", lambda _self: ["c0", "c1", "c2", "c3"])
    monkeypatch.setattr(trainer_module.AudioTrainer, "_ensure_validation_components", lambda _self: None)

    class _FakeSampler:
        def __init__(self, **_kwargs) -> None:
            pass

        def generate(self, config, device) -> types.SimpleNamespace:  # noqa: ANN001, ARG002
            generated_prompts.append(config.prompt)
            return types.SimpleNamespace(
                waveform=torch.ones(1, 32),
                sampling_rate=16_000,
            )

    def _save_stub(output_path: str, _waveform: torch.Tensor, sample_rate: int) -> None:
        saved_calls.append((output_path, sample_rate))

    class _SamplingCtx:
        def start_video(self, _idx: int) -> None:
            return None

        def advance_step(self) -> None:
            return None

        def cleanup(self) -> None:
            return None

    def _start_sampling(*, num_prompts: int, num_steps: int) -> _SamplingCtx:  # noqa: ARG001
        return _SamplingCtx()

    monkeypatch.setattr(trainer_module, "ValidationSampler", _FakeSampler)
    monkeypatch.setattr(trainer_module.torchaudio, "save", _save_stub)

    trainer = trainer_module.AudioTrainer(config)
    trainer._audio_decoder = object()
    trainer._vocoder = object()
    trainer._accelerator = types.SimpleNamespace(
        process_index=1,
        num_processes=2,
        device=torch.device("cpu"),
        unwrap_model=lambda model: model,
    )
    trainer._global_step = 3

    results = trainer._sample_validation_audios(types.SimpleNamespace(start_sampling=_start_sampling))

    assert generated_prompts == ["p1", "p3"]
    assert [item[0] for item in results] == [1, 3]
    assert saved_calls == [
        (str(tmp_path / "samples" / "step_000003_02.wav"), 16_000),
        (str(tmp_path / "samples" / "step_000003_04.wav"), 16_000),
    ]


def test_audio_trainer_offloaded_optimizer_state_moves_cuda_tensors_to_cpu_and_back(
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=True,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir="/tmp/out",
    )

    class _FakeTensor:
        def __init__(self, device: str) -> None:
            self.device = device
            self.is_cuda = device == "cuda"
            self.nbytes = 128
            self.moves: list[str] = []

        def cpu(self) -> "_FakeTensor":
            self.moves.append("cpu")
            return _FakeTensor("cpu")

        def to(self, device: torch.device) -> "_FakeTensor":
            self.moves.append(str(device))
            return _FakeTensor(str(device))

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )

    trainer = trainer_module.AudioTrainer(config)
    fake_tensor = _FakeTensor("cuda")
    trainer._optimizer = types.SimpleNamespace(state={"p": {"exp_avg": fake_tensor}})
    trainer._accelerator = types.SimpleNamespace(device=torch.device("cuda"))

    with trainer._offloaded_optimizer_state():
        assert trainer._optimizer.state["p"]["exp_avg"].device == "cpu"

    assert trainer._optimizer.state["p"]["exp_avg"].device == "cuda"


def test_audio_trainer_offloaded_optimizer_state_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir="/tmp/out",
    )

    class _FakeTensor:
        def __init__(self) -> None:
            self.device = "cuda"
            self.is_cuda = True
            self.nbytes = 128
            self.cpu_calls = 0
            self.to_calls = 0

        def cpu(self) -> "_FakeTensor":
            self.cpu_calls += 1
            return self

        def to(self, _device: torch.device) -> "_FakeTensor":
            self.to_calls += 1
            return self

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )

    trainer = trainer_module.AudioTrainer(config)
    fake_tensor = _FakeTensor()
    trainer._optimizer = types.SimpleNamespace(state={"p": {"exp_avg": fake_tensor}})
    trainer._accelerator = types.SimpleNamespace(device=torch.device("cuda"))

    with trainer._offloaded_optimizer_state():
        assert trainer._optimizer.state["p"]["exp_avg"] is fake_tensor

    assert fake_tensor.cpu_calls == 0
    assert fake_tensor.to_calls == 0


def test_audio_trainer_offloaded_optimizer_state_is_noop_for_fsdp(
    monkeypatch: pytest.MonkeyPatch,
    trainer_module: types.ModuleType,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(training_mode="full", model_path="m", text_encoder_path="t"),
        lora=None,
        optimization=types.SimpleNamespace(
            learning_rate=1e-4,
            optimizer_type="adamw",
            scheduler_type="constant",
            scheduler_params={},
            batch_size=1,
            gradient_accumulation_steps=1,
            steps=1,
            max_grad_norm=1.0,
            enable_gradient_checkpointing=False,
        ),
        acceleration=types.SimpleNamespace(
            mixed_precision_mode="bf16",
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=True,
        ),
        training_strategy=types.SimpleNamespace(name="text_to_audio"),
        data=types.SimpleNamespace(preprocessed_data_root="/tmp/pre", num_dataloader_workers=0),
        validation=types.SimpleNamespace(interval=None, skip_initial_validation=True, prompts=[]),
        checkpoints=types.SimpleNamespace(interval=None, keep_last_n=1, precision="bfloat16", no_resume=True),
        hub=types.SimpleNamespace(push_to_hub=False, hub_model_id=None),
        wandb=types.SimpleNamespace(enabled=False),
        flow_matching=types.SimpleNamespace(
            timestep_sampling_mode="uniform",
            timestep_sampling_params={},
        ),
        seed=42,
        output_dir="/tmp/out",
    )

    class _FakeTensor:
        def __init__(self) -> None:
            self.device = "cuda"
            self.is_cuda = True
            self.nbytes = 128
            self.cpu_calls = 0
            self.to_calls = 0

        def cpu(self) -> "_FakeTensor":
            self.cpu_calls += 1
            return self

        def to(self, _device: torch.device) -> "_FakeTensor":
            self.to_calls += 1
            return self

    monkeypatch.setattr(trainer_module, "print_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "get_training_strategy",
        lambda _cfg: types.SimpleNamespace(get_data_sources=lambda: {"audio_latents": "latent_conditions"}),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_training_components",
        lambda **_kwargs: types.SimpleNamespace(
            transformer=types.SimpleNamespace(parameters=lambda: [parameter]),
            scheduler=None,
            unload_text_encoder=lambda **_kw: None,
        ),
    )

    trainer = trainer_module.AudioTrainer(config)
    fake_tensor = _FakeTensor()
    trainer._optimizer = types.SimpleNamespace(state={"p": {"exp_avg": fake_tensor}})

    class _FakeDistributedType:
        FSDP = "FSDP"

    monkeypatch.setattr(trainer_module, "DistributedType", _FakeDistributedType)
    trainer._accelerator = types.SimpleNamespace(
        distributed_type="FSDP",
        device=torch.device("cuda"),
    )

    with trainer._offloaded_optimizer_state():
        assert trainer._optimizer.state["p"]["exp_avg"] is fake_tensor

    assert fake_tensor.cpu_calls == 0
    assert fake_tensor.to_calls == 0
