"""Weight sources for block streaming: protocol and implementations."""

from __future__ import annotations

from collections import OrderedDict
from typing import Protocol

import torch

from ltx_audio_core.block_streaming.disk import DiskBlockReader
from ltx_audio_core.block_streaming.pool import WeightPool
from ltx_audio_core.loader.primitives import TensorLayout


class WeightSource(Protocol):
    """Provides pinned CPU weights for a given block index."""

    @property
    def block_layout(self) -> TensorLayout: ...

    def get(self, idx: int) -> dict[str, torch.Tensor]: ...

    def release(self, idx: int, event: torch.cuda.Event) -> None: ...

    def cleanup(self) -> None: ...


class DiskWeightSource(WeightSource):
    """Reads block weights from disk into pinned CPU buffers on demand."""

    def __init__(self, pool: WeightPool, reader: DiskBlockReader) -> None:
        self._pool = pool
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._events: dict[int, torch.cuda.Event] = {}
        self._reader = reader

    @property
    def block_layout(self) -> TensorLayout:
        return self._pool.buffer_layout

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        if idx in self._cache:
            return self._cache[idx]

        if len(self._cache) >= self._pool.capacity:
            evicted_idx, evicted_weights = self._cache.popitem(last=False)
            self._pool.release(evicted_weights, event=self._events.pop(evicted_idx, None))

        weights = self._pool.acquire()
        self._reader.read_into(weights, idx)
        self._cache[idx] = weights
        return weights

    def release(self, idx: int, event: torch.cuda.Event) -> None:
        self._events[idx] = event

    def cleanup(self) -> None:
        self._cache.clear()
        self._events.clear()
        self._reader.cleanup()

    def __len__(self) -> int:
        return len(self._cache)


class PinnedWeightSource(WeightSource):
    """Pre-loaded pinned CPU weights."""

    def __init__(self, weights: dict[int, dict[str, torch.Tensor]]) -> None:
        if not weights:
            raise ValueError("PinnedWeightSource requires at least one block")
        self._weights = weights

    @property
    def block_layout(self) -> TensorLayout:
        first_block = self._weights[min(self._weights)]
        return {name: (t.shape, t.dtype) for name, t in first_block.items()}

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        return self._weights[idx]

    def release(self, idx: int, event: torch.cuda.Event) -> None:
        pass

    def cleanup(self) -> None:
        self._weights.clear()

    def __len__(self) -> int:
        return len(self._weights)
