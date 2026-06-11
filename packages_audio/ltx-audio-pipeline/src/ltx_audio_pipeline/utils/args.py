from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ltx_audio_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_audio_core.model.transformer.compiling import CompilationConfig
from ltx_audio_pipeline.utils.constants import LTX_2_3_PARAMS, PipelineParams
from ltx_audio_pipeline.utils.quantization_factory import QuantizationKind
from ltx_audio_pipeline.utils.types import OffloadMode

DEFAULT_NEGATIVE_PROMPT = (
    "distorted, noisy, muffled, clipped, robotic, metallic resonance, harsh transients, unstable pitch, "
    "unnatural timing, silence gaps, repetitive artifacts, background hiss, low fidelity, AI artifacts"
)
DEFAULT_LORA_STRENGTH = 1.0
QUANTIZATION_POLICIES = tuple(kind.value for kind in QuantizationKind)


class LoraAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) > 2:
            msg = f"{option_string} accepts at most 2 arguments (PATH and optional STRENGTH), got {len(values)} values"
            raise argparse.ArgumentError(self, msg)

        resolved_path = resolve_existing_path(values[0])
        strength = float(values[1]) if len(values) == 2 else DEFAULT_LORA_STRENGTH

        current = getattr(namespace, self.dest) or []
        current.append(LoraPathStrengthAndSDOps(resolved_path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
        setattr(namespace, self.dest, current)


class CompileAction(argparse.Action):
    _ALLOWED_KEYS = frozenset({"mode", "backend", "fullgraph", "dynamic", "inductor_config", "dynamo_config"})

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,  # noqa: ARG002
    ) -> None:
        overrides: dict[str, object] = {}
        for item in values:
            if "=" not in item:
                raise argparse.ArgumentError(self, f"expects KEY=VALUE pairs, got: {item!r}")
            key, _, raw = item.partition("=")
            key = key.strip()
            if key not in self._ALLOWED_KEYS:
                raise argparse.ArgumentError(
                    self,
                    f"{key!r} is not a CompilationConfig field; valid keys: {sorted(self._ALLOWED_KEYS)}",
                )
            if key in overrides:
                raise argparse.ArgumentError(self, f"{key} given more than once")
            if key == "mode":
                overrides[key] = self._parse_mode(raw)
            elif key == "backend":
                overrides[key] = self._parse_non_empty(key, raw)
            elif key == "fullgraph":
                overrides[key] = self._parse_bool(key, raw)
            elif key == "dynamic":
                overrides[key] = self._parse_dynamic(raw)
            elif key in ("inductor_config", "dynamo_config"):
                overrides[key] = self._parse_json_dict(key, raw)
        setattr(namespace, self.dest, CompilationConfig(**overrides))

    def _parse_mode(self, raw: str) -> str | None:
        stripped = raw.strip()
        if not stripped:
            raise argparse.ArgumentError(self, "mode=... value cannot be empty (use mode=none to clear)")
        if stripped.lower() == "none":
            return None
        return stripped

    def _parse_non_empty(self, key: str, raw: str) -> str:
        stripped = raw.strip()
        if not stripped:
            raise argparse.ArgumentError(self, f"{key}=... value cannot be empty")
        return stripped

    def _parse_bool(self, key: str, raw: str) -> bool:
        normalized = raw.strip().lower()
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False
        raise argparse.ArgumentError(self, f"{key}=... must be true or false; got {raw!r}")

    def _parse_dynamic(self, raw: str) -> bool | None:
        normalized = raw.strip().lower()
        if normalized in ("auto", "none"):
            return None
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False
        raise argparse.ArgumentError(self, f"dynamic=... must be auto/true/false; got {raw!r}")

    def _parse_json_dict(self, key: str, raw: str) -> dict[str, Any]:
        stripped = raw.strip()
        if not stripped:
            raise argparse.ArgumentError(self, f"{key}=... value cannot be empty")
        if stripped.startswith("{"):
            source = stripped
        else:
            path = Path(stripped).expanduser()
            if not path.is_file():
                raise argparse.ArgumentError(
                    self, f"{key}=... must be a JSON object or a path to a JSON file; got {raw!r}"
                )
            source = path.read_text()
        try:
            value = json.loads(source)
        except json.JSONDecodeError as e:
            raise argparse.ArgumentError(self, f"{key}=... must be a JSON object; got {raw!r} ({e.msg})") from None
        if not isinstance(value, dict):
            raise argparse.ArgumentError(self, f"{key}=... must decode to a JSON object; got {type(value).__name__}")
        return value


def resolve_path(path: str) -> str:
    return str(Path(path).expanduser().resolve().as_posix())


def resolve_existing_path(path: str) -> str:
    resolved = resolve_path(path)
    if not Path(resolved).exists():
        raise argparse.ArgumentError(None, f"Path not found: {resolved}")
    return resolved


def _resolve_quantization(namespace: argparse.Namespace) -> None:
    name = getattr(namespace, "quantization", None)
    if name is None:
        return
    if not isinstance(name, str):
        return
    namespace.quantization = QuantizationKind(name).to_policy(checkpoint_path=namespace.checkpoint_path)


class _PipelineArgumentParser(argparse.ArgumentParser):
    def parse_args(  # type: ignore[override]
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        ns = super().parse_args(args, namespace)
        if ns.compile is not None and ns.offload_mode != OffloadMode.NONE:
            self.error("--compile cannot be combined with --offload")
        _resolve_quantization(ns)
        return ns


def detect_checkpoint_path() -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--checkpoint-path", type=resolve_existing_path, required=True)
    known, _ = pre.parse_known_args()
    return known.checkpoint_path


def basic_arg_parser(params: PipelineParams = LTX_2_3_PARAMS) -> argparse.ArgumentParser:
    parser = _PipelineArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=resolve_existing_path,
        required=True,
        help="Path to the LTX audio checkpoint (.safetensors file).",
    )
    parser.add_argument(
        "--gemma-root",
        type=resolve_existing_path,
        required=True,
        help="Path to the root directory containing the Gemma text encoder files.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt describing the target audio content.")
    parser.add_argument(
        "--output-path",
        type=resolve_path,
        required=True,
        help="Path to the output audio file (WAV format).",
    )
    parser.add_argument("--seed", type=int, default=params.seed, help="Random seed for reproducible generation.")
    parser.add_argument(
        "--lora",
        dest="lora",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="LoRA path and optional strength. Can be specified multiple times.",
    )
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument(
        "--offload",
        dest="offload_mode",
        type=OffloadMode,
        default=OffloadMode.NONE,
        choices=list(OffloadMode),
        help="Weight offloading strategy for layer streaming (none/cpu/disk). Incompatible with --compile.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Maximum batch size per transformer forward pass.",
    )
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_POLICIES,
        default=None,
        help=f"Quantization policy: {', '.join(QUANTIZATION_POLICIES)}.",
    )
    parser.add_argument(
        "--compile",
        nargs="*",
        action=CompileAction,
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Enable torch.compile for transformer blocks. Pass alone for defaults, "
            "or with KEY=VALUE overrides for any CompilationConfig field. Incompatible with --offload."
        ),
    )
    return parser


def audio_one_stage_arg_parser(params: PipelineParams = LTX_2_3_PARAMS) -> argparse.ArgumentParser:
    parser = basic_arg_parser(params=params)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt describing what should be avoided in the generated audio.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=params.height,
        help="Reference canvas height used to derive audio duration.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=params.width,
        help="Reference canvas width used to derive audio duration.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=params.num_frames,
        help="Reference frame count used to derive the target audio duration.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=params.frame_rate,
        help="Reference frame rate used to derive duration.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=params.num_inference_steps, help="Number of denoising steps.")
    parser.add_argument(
        "--audio-cfg-guidance-scale",
        type=float,
        default=params.audio_guider_params.cfg_scale,
        help="Audio CFG guidance scale.",
    )
    parser.add_argument(
        "--audio-stg-guidance-scale",
        type=float,
        default=params.audio_guider_params.stg_scale,
        help="Audio STG guidance scale.",
    )
    parser.add_argument(
        "--audio-rescale-scale",
        type=float,
        default=params.audio_guider_params.rescale_scale,
        help="Audio rescale guidance scale.",
    )
    parser.add_argument(
        "--audio-stg-blocks",
        type=int,
        nargs="*",
        default=params.audio_guider_params.stg_blocks,
        help="Which transformer blocks to perturb for audio STG.",
    )
    parser.add_argument(
        "--audio-skip-step",
        type=int,
        default=params.audio_guider_params.skip_step,
        help="Periodic skip factor for audio denoising.",
    )
    parser.add_argument(
        "--v2a-guidance-scale",
        type=float,
        default=params.audio_guider_params.modality_scale,
        help="Video-to-audio guidance scale.",
    )
    return parser


__all__ = [
    "audio_one_stage_arg_parser",
    "basic_arg_parser",
    "detect_checkpoint_path",
    "resolve_existing_path",
    "resolve_path",
]
