from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ONE_STAGE_PATH = PACKAGE_ROOT / "src" / "ltx_audio_pipeline" / "audio_one_stage.py"
ARGS_PATH = PACKAGE_ROOT / "src" / "ltx_audio_pipeline" / "utils" / "args.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_audio_one_stage_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = object()
    torch_module.float32 = object()
    torch_module.Tensor = object
    torch_module.device = str

    class _Generator:
        def __init__(self, device: object | None = None) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> "_Generator":
            self.seed = seed
            return self

    def _inference_mode():
        def decorator(fn):
            return fn

        return decorator

    torch_module.Generator = _Generator
    torch_module.inference_mode = _inference_mode
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    torchaudio_module = types.ModuleType("torchaudio")
    torchaudio_module.save = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_module)

    guiders_module = types.ModuleType("ltx_audio_core.components.guiders")

    @dataclass
    class MultiModalGuiderParams:
        cfg_scale: float
        stg_scale: float
        rescale_scale: float
        modality_scale: float
        skip_step: int
        stg_blocks: list[int]

    class MultiModalGuiderFactory:
        pass

    guiders_module.MultiModalGuiderFactory = MultiModalGuiderFactory
    guiders_module.MultiModalGuiderParams = MultiModalGuiderParams
    guiders_module.create_multimodal_guider_factory = lambda params, negative_context: (
        params,
        negative_context,
    )
    monkeypatch.setitem(sys.modules, "ltx_audio_core.components.guiders", guiders_module)

    noisers_module = types.ModuleType("ltx_audio_core.components.noisers")
    noisers_module.GaussianNoiser = type("GaussianNoiser", (), {"__init__": lambda self, generator: None})
    monkeypatch.setitem(sys.modules, "ltx_audio_core.components.noisers", noisers_module)

    schedulers_module = types.ModuleType("ltx_audio_core.components.schedulers")
    schedulers_module.LTX2Scheduler = type("LTX2Scheduler", (), {})
    monkeypatch.setitem(sys.modules, "ltx_audio_core.components.schedulers", schedulers_module)

    loader_primitives = types.ModuleType("ltx_audio_core.loader.primitives")
    loader_primitives.LoraPathStrengthAndSDOps = tuple
    monkeypatch.setitem(sys.modules, "ltx_audio_core.loader.primitives", loader_primitives)

    loader_registry = types.ModuleType("ltx_audio_core.loader.registry")
    loader_registry.Registry = type("Registry", (), {})
    monkeypatch.setitem(sys.modules, "ltx_audio_core.loader.registry", loader_registry)

    compiling_module = types.ModuleType("ltx_audio_core.model.transformer.compiling")
    compiling_module.CompilationConfig = type("CompilationConfig", (), {})
    monkeypatch.setitem(sys.modules, "ltx_audio_core.model.transformer.compiling", compiling_module)

    quantization_module = types.ModuleType("ltx_audio_core.quantization")
    quantization_module.QuantizationPolicy = type("QuantizationPolicy", (), {})
    monkeypatch.setitem(sys.modules, "ltx_audio_core.quantization", quantization_module)

    types_module = types.ModuleType("ltx_audio_core.types")
    types_module.Audio = type("Audio", (), {})
    monkeypatch.setitem(sys.modules, "ltx_audio_core.types", types_module)

    utils_module = types.ModuleType("ltx_audio_pipeline.utils")

    class OffloadMode(Enum):
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    class ModalitySpec:
        def __init__(self, context=None) -> None:
            self.context = context

    utils_module.AudioDecoder = type("AudioDecoder", (), {})
    utils_module.DiffusionStage = type("DiffusionStage", (), {})
    utils_module.FactoryGuidedDenoiser = type("FactoryGuidedDenoiser", (), {})
    utils_module.ModalitySpec = ModalitySpec
    utils_module.OffloadMode = OffloadMode
    utils_module.PromptEncoder = type("PromptEncoder", (), {})
    utils_module.SimpleDenoiser = type("SimpleDenoiser", (), {})
    utils_module.get_device = lambda: "cpu"
    monkeypatch.setitem(sys.modules, "ltx_audio_pipeline.utils", utils_module)

    args_module = types.ModuleType("ltx_audio_pipeline.utils.args")

    args_namespace = argparse.Namespace(
        checkpoint_path="/tmp/checkpoint.safetensors",
        gemma_root="/tmp/gemma",
        lora=[("lora-a.safetensors", 0.5), ("lora-b.safetensors", 1.0)],
        quantization="fp8_scaled",
        compile="compile-config",
        offload_mode=OffloadMode.CPU,
        prompt="soft piano",
        negative_prompt="noise",
        seed=7,
        height=256,
        width=512,
        num_frames=65,
        frame_rate=16.0,
        num_inference_steps=18,
        audio_cfg_guidance_scale=6.5,
        audio_stg_guidance_scale=1.25,
        audio_rescale_scale=0.6,
        v2a_guidance_scale=2.5,
        audio_skip_step=3,
        audio_stg_blocks=[4, 8],
        enhance_prompt=True,
        max_batch_size=2,
        output_path="/tmp/out.wav",
    )

    class _Parser:
        def parse_args(self):
            captured["parse_args_called"] = True
            return args_namespace

    def _audio_one_stage_arg_parser(*, params):
        captured["parser_params"] = params
        return _Parser()

    args_module.audio_one_stage_arg_parser = _audio_one_stage_arg_parser
    args_module.detect_checkpoint_path = lambda: "/tmp/checkpoint.safetensors"
    monkeypatch.setitem(sys.modules, "ltx_audio_pipeline.utils.args", args_module)

    constants_module = types.ModuleType("ltx_audio_pipeline.utils.constants")
    constants_module.detect_params = lambda checkpoint_path: {"checkpoint_path": checkpoint_path, "kind": "stub"}
    monkeypatch.setitem(sys.modules, "ltx_audio_pipeline.utils.constants", constants_module)

    return captured


def _install_args_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    loader_module = types.ModuleType("ltx_audio_core.loader")
    loader_module.LTXV_LORA_COMFY_RENAMING_MAP = {"stub": "stub"}

    @dataclass
    class LoraPathStrengthAndSDOps:
        path: str
        strength: float
        sd_ops: object

    loader_module.LoraPathStrengthAndSDOps = LoraPathStrengthAndSDOps
    monkeypatch.setitem(sys.modules, "ltx_audio_core.loader", loader_module)

    compiling_module = types.ModuleType("ltx_audio_core.model.transformer.compiling")

    @dataclass
    class CompilationConfig:
        mode: str | None = "default"
        backend: str = "inductor"
        fullgraph: bool = False
        dynamic: bool | None = None
        inductor_config: dict[str, object] = field(default_factory=dict)
        dynamo_config: dict[str, object] = field(default_factory=dict)

    compiling_module.CompilationConfig = CompilationConfig
    monkeypatch.setitem(sys.modules, "ltx_audio_core.model.transformer.compiling", compiling_module)

    constants_module = types.ModuleType("ltx_audio_pipeline.utils.constants")

    @dataclass(frozen=True)
    class _GuiderDefaults:
        cfg_scale: float = 7.0
        stg_scale: float = 1.0
        rescale_scale: float = 0.7
        modality_scale: float = 3.0
        skip_step: int = 0
        stg_blocks: list[int] = field(default_factory=lambda: [28])

    @dataclass(frozen=True)
    class PipelineParams:
        seed: int = 10
        height: int = 512
        width: int = 768
        num_frames: int = 121
        frame_rate: float = 24.0
        num_inference_steps: int = 30
        audio_guider_params: _GuiderDefaults = field(default_factory=_GuiderDefaults)

    constants_module.LTX_2_3_PARAMS = PipelineParams()
    constants_module.PipelineParams = PipelineParams
    monkeypatch.setitem(sys.modules, "ltx_audio_pipeline.utils.constants", constants_module)

    quantization_module = types.ModuleType("ltx_audio_pipeline.utils.quantization_factory")

    class QuantizationKind(Enum):
        FP8_SCALED = "fp8_scaled"
        BF16 = "bf16"

        def to_policy(self, checkpoint_path: str):
            return (self.value, checkpoint_path)

    quantization_module.QuantizationKind = QuantizationKind
    monkeypatch.setitem(sys.modules, "ltx_audio_pipeline.utils.quantization_factory", quantization_module)

    types_module = types.ModuleType("ltx_audio_pipeline.utils.types")

    class OffloadMode(Enum):
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    types_module.OffloadMode = OffloadMode
    monkeypatch.setitem(sys.modules, "ltx_audio_pipeline.utils.types", types_module)


def test_audio_one_stage_main_wires_cli_args_into_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_audio_one_stage_stubs(monkeypatch)
    module = _load_module("audio_one_stage_entry_test", AUDIO_ONE_STAGE_PATH)

    fake_audio = object()
    pipeline_init: dict[str, object] = {}
    pipeline_call: dict[str, object] = {}
    save_call: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, **kwargs) -> None:
            pipeline_init.update(kwargs)

        def __call__(self, **kwargs):
            pipeline_call.update(kwargs)
            return fake_audio

    monkeypatch.setattr(module, "AudioOneStagePipeline", FakePipeline)
    monkeypatch.setattr(module, "_save_audio", lambda audio, output_path: save_call.update(audio=audio, output_path=output_path))

    module.main()

    assert captured["parse_args_called"] is True
    assert captured["parser_params"] == {"checkpoint_path": "/tmp/checkpoint.safetensors", "kind": "stub"}
    assert pipeline_init == {
        "checkpoint_path": "/tmp/checkpoint.safetensors",
        "gemma_root": "/tmp/gemma",
        "loras": (("lora-a.safetensors", 0.5), ("lora-b.safetensors", 1.0)),
        "quantization": "fp8_scaled",
        "compilation_config": "compile-config",
        "offload_mode": module.OffloadMode.CPU,
    }
    assert pipeline_call["prompt"] == "soft piano"
    assert pipeline_call["negative_prompt"] == "noise"
    assert pipeline_call["seed"] == 7
    assert pipeline_call["height"] == 256
    assert pipeline_call["width"] == 512
    assert pipeline_call["num_frames"] == 65
    assert pipeline_call["frame_rate"] == 16.0
    assert pipeline_call["num_inference_steps"] == 18
    assert pipeline_call["enhance_prompt"] is True
    assert pipeline_call["max_batch_size"] == 2
    guider_params = pipeline_call["audio_guider_params"]
    assert guider_params.cfg_scale == 6.5
    assert guider_params.stg_scale == 1.25
    assert guider_params.rescale_scale == 0.6
    assert guider_params.modality_scale == 2.5
    assert guider_params.skip_step == 3
    assert guider_params.stg_blocks == [4, 8]
    assert save_call == {"audio": fake_audio, "output_path": "/tmp/out.wav"}


def test_basic_arg_parser_rejects_compile_and_offload_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_args_stubs(monkeypatch)
    module = _load_module("audio_args_entry_test", ARGS_PATH)

    checkpoint_path = tmp_path / "checkpoint.safetensors"
    checkpoint_path.write_text("stub", encoding="utf-8")
    gemma_root = tmp_path / "gemma"
    gemma_root.mkdir()

    parser = module.basic_arg_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "--checkpoint-path",
                str(checkpoint_path),
                "--gemma-root",
                str(gemma_root),
                "--prompt",
                "soft piano",
                "--output-path",
                str(tmp_path / "out.wav"),
                "--offload",
                "cpu",
                "--compile",
            ]
        )

    assert exc_info.value.code == 2
