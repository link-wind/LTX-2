from dataclasses import dataclass, field
from typing import Any

import torch

from ltx_audio_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_audio_core.loader.module_ops import ModuleOps
from ltx_audio_core.loader.sd_ops import SDOps
from ltx_audio_core.model.transformer.model import LTXModel
from ltx_audio_core.model.transformer.transformer_args import BlockPerturbationsProcessor, TransformerArgs

_DEFAULT_INDUCTOR_CONFIG: dict[str, Any] = {"unsafe_skip_cache_dynamic_shape_guards": True}
_DEFAULT_DYNAMO_CONFIG: dict[str, Any] = {"inline_inbuilt_nn_modules": True, "cache_size_limit": 256}


@dataclass(frozen=True)
class CompilationConfig:
    """``torch.compile`` configuration for transformer blocks. ``None`` keeps eager."""

    mode: str | None = None
    backend: str = "inductor"
    fullgraph: bool = False
    dynamic: bool | None = None
    inductor_config: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_INDUCTOR_CONFIG))
    dynamo_config: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_DYNAMO_CONFIG))


class _SeqDynamicMarkingProcessor:
    """Marks the per-block seq dim dynamic, then delegates to an inner processor."""

    def __init__(self, inner: BlockPerturbationsProcessor) -> None:
        self.inner = inner

    def __call__(
        self,
        args: TransformerArgs,
        perturbations: BatchedPerturbationConfig,
        block_idx: int,
        self_attn_type: PerturbationType,
        cross_attn_type: PerturbationType,
    ) -> TransformerArgs:
        torch._dynamo.mark_dynamic(args.x, 1)
        cos, sin = args.positional_embeddings
        torch._dynamo.mark_dynamic(cos, cos.ndim - 2)
        torch._dynamo.mark_dynamic(sin, sin.ndim - 2)
        if args.cross_positional_embeddings is not None:
            cross_cos, cross_sin = args.cross_positional_embeddings
            torch._dynamo.mark_dynamic(cross_cos, cross_cos.ndim - 2)
            torch._dynamo.mark_dynamic(cross_sin, cross_sin.ndim - 2)
        if args.self_attention_mask is not None:
            if args.self_attention_mask.shape[2] > 1:
                torch._dynamo.mark_dynamic(args.self_attention_mask, 2)
            torch._dynamo.mark_dynamic(args.self_attention_mask, 3)
        if args.context_mask is not None:
            torch._dynamo.mark_dynamic(args.context_mask, 2)
        if args.timesteps.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.timesteps, 1)
        if args.embedded_timestep.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.embedded_timestep, 1)
        if args.cross_scale_shift_timestep is not None and args.cross_scale_shift_timestep.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.cross_scale_shift_timestep, 1)
        return self.inner(args, perturbations, block_idx, self_attn_type, cross_attn_type)


def compile_transformer(model: LTXModel, config: CompilationConfig) -> LTXModel:
    model.transformer_blocks = torch.nn.ModuleList(
        torch.compile(m, mode=config.mode, backend=config.backend, fullgraph=config.fullgraph, dynamic=config.dynamic)
        for m in model.transformer_blocks
    )
    model.block_input_processor = _SeqDynamicMarkingProcessor(inner=model.block_input_processor)

    def patched_dynamo_forward(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        torch.compiler.cudagraph_mark_step_begin()
        with (
            torch._inductor.config.patch(**config.inductor_config),
            torch._dynamo.config.patch(**config.dynamo_config),  # type: ignore[attr-defined]
        ):
            return model.forward_without_compilation(*args, **kwargs)

    model.forward_without_compilation = model.forward
    model.forward = patched_dynamo_forward
    return model


def build_compile_transformer_op(config: CompilationConfig) -> ModuleOps:
    return ModuleOps(
        name="compile_transformer",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=lambda model: compile_transformer(model, config),
    )


def modify_sd_ops_for_compilation(original_sd_ops: SDOps, number_of_blocks: int = 48) -> SDOps:
    for i in range(number_of_blocks):
        original_sd_ops = original_sd_ops.with_replacement(
            f"transformer_blocks.{i}.", f"transformer_blocks.{i}._orig_mod."
        )
    return original_sd_ops
