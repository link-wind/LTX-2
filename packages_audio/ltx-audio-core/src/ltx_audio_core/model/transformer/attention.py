import functools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ltx_audio_core.model.transformer.ops import (
    GatedAttentionCallable,
    PreAttentionCallable,
    PytorchGatedAttention,
    PytorchPreAttention,
)
from ltx_audio_core.model.transformer.rope import LTXRopeType

logger = logging.getLogger(__name__)


def _torch_default_sdpa_priority() -> list[SDPBackend]:
    """Fetch torch's current default SDPA priority order at runtime."""
    return [SDPBackend(p) for p in torch._C._get_sdp_priority_order()]


memory_efficient_attention = None
flash_attn_interface = None
flash_attn_4_func = None

try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

try:
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None

try:
    from flash_attn.cute import flash_attn_func as flash_attn_4_func
except ImportError:
    flash_attn_4_func = None


class AttentionCallable(Protocol):
    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor: ...


class MaskedAttentionCallable(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor,
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __init__(self, priority: list[SDPBackend] | None = None) -> None:
        self._priority = priority if priority is not None else _torch_default_sdpa_priority()

    @property
    def label(self) -> str:
        return f"SDPA[{'>'.join(b.name for b in self._priority)}]"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        with sdpa_kernel(self._priority, set_priority=True):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    label = "xFormers"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            pad = 8 - mask.shape[-1] % 8
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad],
                dtype=q.dtype,
                device=q.device,
            )

            mask_out[..., : mask.shape[-1]] = mask
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    label = "FlashAttention3"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention4(AttentionCallable):
    label = "FlashAttention4"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        if flash_attn_4_func is None:
            raise RuntimeError("FlashAttention4 was selected but `flash-attn-4` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        out, _ = flash_attn_4_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


def _sdpa_can_use(backend: SDPBackend, *, with_mask: bool) -> bool:
    if backend is SDPBackend.MATH:
        return True
    if not torch.cuda.is_available():
        return False

    q = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(1, 4, 128, 128, device="cuda", dtype=torch.bfloat16) if with_mask else None
    params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, False, False)

    if backend is SDPBackend.CUDNN_ATTENTION:
        return torch.backends.cuda.can_use_cudnn_attention(params, debug=False)
    if backend is SDPBackend.FLASH_ATTENTION:
        return torch.backends.cuda.can_use_flash_attention(params, debug=False)
    if backend is SDPBackend.EFFICIENT_ATTENTION:
        return torch.backends.cuda.can_use_efficient_attention(params, debug=False)
    return False


_SDPA_FULL_PRIORITY: tuple[SDPBackend, ...] = (
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)


def _sdpa_full_priority() -> PytorchAttention:
    return PytorchAttention(priority=list(_SDPA_FULL_PRIORITY))


def _select_primary_attention() -> AttentionCallable:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major == 9:
            if flash_attn_interface is not None:
                return FlashAttention3()
            if memory_efficient_attention is not None:
                return XFormersAttention()
            if flash_attn_4_func is not None:
                return FlashAttention4()
        if major == 10 and flash_attn_4_func is not None:
            return FlashAttention4()
    return _sdpa_full_priority()


def _select_masked_attention() -> MaskedAttentionCallable:
    if memory_efficient_attention is not None:
        return XFormersAttention()
    return _sdpa_full_priority()


@functools.cache
def automatic_attention() -> AttentionCallable:
    fn = _select_primary_attention()
    logger.info("Automatic attention selected: %s", fn.label)
    return fn


@functools.cache
def automatic_masked_attention() -> MaskedAttentionCallable:
    fn = _select_masked_attention()
    logger.info("Automatic masked attention selected: %s", fn.label)
    return fn


def _resolve_sdpa_variant(backend: SDPBackend, name: str, *, with_mask: bool) -> PytorchAttention:
    if not _sdpa_can_use(backend, with_mask=with_mask):
        raise RuntimeError(
            f"{name} selected but the SDPA {backend.name} backend is not usable on this machine "
            "(either no CUDA, the backend rejected the probe shapes, or "
            "torch.use_deterministic_algorithms(True) excluded it)."
        )
    return PytorchAttention(priority=[backend])


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_4 = "flash_attention_4"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_FLASH = "sdpa_flash"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
    AUTOMATIC = "automatic"

    def to_callable(self) -> AttentionCallable:  # noqa: PLR0911
        match self:
            case AttentionFunction.AUTOMATIC:
                return automatic_attention()
            case AttentionFunction.PYTORCH:
                return PytorchAttention()
            case AttentionFunction.XFORMERS:
                if memory_efficient_attention is None:
                    raise RuntimeError("AttentionFunction.XFORMERS selected but `xformers` is not installed.")
                return XFormersAttention()
            case AttentionFunction.FLASH_ATTENTION_3:
                if flash_attn_interface is None:
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_3 selected but `flash-attn-3` is not installed."
                    )
                return FlashAttention3()
            case AttentionFunction.FLASH_ATTENTION_4:
                if flash_attn_4_func is None:
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_4 selected but `flash-attn-4` is not installed."
                    )
                return FlashAttention4()
            case AttentionFunction.SDPA_MATH:
                return PytorchAttention(priority=[SDPBackend.MATH])
            case AttentionFunction.SDPA_CUDNN:
                return _resolve_sdpa_variant(
                    SDPBackend.CUDNN_ATTENTION, "AttentionFunction.SDPA_CUDNN", with_mask=False
                )
            case AttentionFunction.SDPA_FLASH:
                return _resolve_sdpa_variant(
                    SDPBackend.FLASH_ATTENTION, "AttentionFunction.SDPA_FLASH", with_mask=False
                )
            case AttentionFunction.SDPA_EFFICIENT:
                return _resolve_sdpa_variant(
                    SDPBackend.EFFICIENT_ATTENTION, "AttentionFunction.SDPA_EFFICIENT", with_mask=False
                )


class MaskedAttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
    AUTOMATIC = "automatic"

    def to_callable(self) -> MaskedAttentionCallable:
        match self:
            case MaskedAttentionFunction.AUTOMATIC:
                return automatic_masked_attention()
            case MaskedAttentionFunction.PYTORCH:
                return PytorchAttention()
            case MaskedAttentionFunction.XFORMERS:
                if memory_efficient_attention is None:
                    raise RuntimeError("MaskedAttentionFunction.XFORMERS selected but `xformers` is not installed.")
                return XFormersAttention()
            case MaskedAttentionFunction.SDPA_MATH:
                return PytorchAttention(priority=[SDPBackend.MATH])
            case MaskedAttentionFunction.SDPA_CUDNN:
                return _resolve_sdpa_variant(
                    SDPBackend.CUDNN_ATTENTION, "MaskedAttentionFunction.SDPA_CUDNN", with_mask=True
                )
            case MaskedAttentionFunction.SDPA_EFFICIENT:
                return _resolve_sdpa_variant(
                    SDPBackend.EFFICIENT_ATTENTION, "MaskedAttentionFunction.SDPA_EFFICIENT", with_mask=True
                )


@dataclass(frozen=True)
class AttentionOps:
    attention_function: AttentionCallable = field(default_factory=lambda: AttentionFunction.AUTOMATIC.to_callable())
    masked_attention_function: MaskedAttentionCallable = field(
        default_factory=lambda: MaskedAttentionFunction.AUTOMATIC.to_callable()
    )
    preattention_function: PreAttentionCallable = field(default_factory=PytorchPreAttention)
    gated_attention_function: GatedAttentionCallable = field(default_factory=PytorchGatedAttention)


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        ops: AttentionOps | None = None,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        if ops is None:
            ops = AttentionOps()
        self.rope_type = rope_type
        self.attention_function = ops.attention_function
        self.masked_attention_function = ops.masked_attention_function
        self.preattention_function = ops.preattention_function
        self.gated_attention_function = ops.gated_attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)
            q, k = self.preattention_function(q, k, self, mask, pe, k_pe)
            if mask is None:
                out = self.attention_function(q, k, v, self.heads)
            else:
                out = self.masked_attention_function(q, k, v, self.heads, mask)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            out = self.gated_attention_function(x, out, self)

        return self.to_out(out)
