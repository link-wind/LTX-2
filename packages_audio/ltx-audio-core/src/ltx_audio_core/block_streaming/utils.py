"""Shared utilities for the block_streaming package."""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ltx_audio_core.loader.primitives import TensorLayout

FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
_BUFFER_ALIGN = 16


def make_block_key(blocks_prefix: str, block_idx: int, param_name: str) -> str:
    return f"{blocks_prefix}.{block_idx}.{param_name}"


def resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


def assign_tensor_to_module(root: nn.Module, dotted_name: str, tensor: torch.Tensor) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf in parent._parameters:
        parent._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    elif leaf in parent._buffers:
        parent._buffers[leaf] = tensor
    else:
        raise AttributeError(f"{leaf} is not a parameter or buffer of {type(parent).__name__}")


def derive_layout(tensors: dict[str, torch.Tensor], dtype: torch.dtype | None = None) -> TensorLayout:
    return {
        name: (t.shape, t.dtype if dtype is None or t.dtype in FP8_DTYPES else dtype) for name, t in tensors.items()
    }


def _align_up(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


def _alloc_pinned_exact(nbytes: int) -> torch.Tensor | None:
    cudart = torch.cuda.cudart()
    buf = torch.empty(nbytes, dtype=torch.uint8)
    ptr = buf.data_ptr()
    err = int(cudart.cudaHostRegister(ptr, nbytes, 0))
    if err != 0:
        return None
    weakref.finalize(buf.untyped_storage(), lambda p=ptr: cudart.cudaHostUnregister(p))
    return buf


def _alloc_buffer(nbytes: int, device: torch.device | None, pin_memory: bool) -> torch.Tensor:
    if pin_memory and (device is None or torch.device(device).type == "cpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("pin_memory=True requires CUDA, which is not available")
        buf = _alloc_pinned_exact(nbytes)
        if buf is not None:
            return buf
    return torch.empty(nbytes, dtype=torch.uint8, device=device, pin_memory=pin_memory)


@dataclass(frozen=True)
class _TensorSlice:
    offset: int
    shape: torch.Size
    dtype: torch.dtype

    def size(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


def allocate_layout_views(
    layout: TensorLayout,
    device: torch.device | None = None,
    pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    slices: dict[str, _TensorSlice] = {}
    cursor = 0
    for key, (shape, dtype) in layout.items():
        cursor = _align_up(cursor, _BUFFER_ALIGN)
        slices[key] = _TensorSlice(offset=cursor, shape=shape, dtype=dtype)
        cursor += slices[key].size()
    buffer = _alloc_buffer(max(_align_up(cursor, _BUFFER_ALIGN), 1), device, pin_memory)
    return {key: buffer[s.offset : s.offset + s.size()].view(s.dtype).view(s.shape) for key, s in slices.items()}
