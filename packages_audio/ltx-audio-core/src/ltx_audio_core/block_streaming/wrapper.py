"""Block streaming wrapper: streams transformer blocks through a WeightsProvider."""

from __future__ import annotations

import itertools
from typing import Any

import torch
from torch import nn

from ltx_audio_core.block_streaming.provider import WeightsProvider
from ltx_audio_core.block_streaming.utils import assign_tensor_to_module


class BlockStreamingWrapper(nn.Module):
    """Streams sequential model blocks through GPU buffer caches."""

    def __init__(
        self,
        model: nn.Module,
        blocks: nn.ModuleList,
        provider: WeightsProvider,
        target_device: torch.device,
    ) -> None:
        super().__init__()
        self._model = model
        self._blocks = blocks
        self._target_device = target_device
        self._provider = provider

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _pre_hook(self, block_idx: int) -> None:
        gpu_weights = self._provider.get(block_idx)

        block = self._blocks[block_idx]
        for name, _param in itertools.chain(block.named_parameters(), block.named_buffers()):
            assign_tensor_to_module(block, name, gpu_weights[name])

    def _post_hook(self, block_idx: int) -> None:
        compute_done = torch.cuda.Event()
        compute_done.record(torch.cuda.current_stream(self._target_device))
        self._provider.release(block_idx, event=compute_done)

    def _register_hooks(self) -> None:
        for idx, block in enumerate(self._blocks):
            pre = block.register_forward_pre_hook(
                lambda _mod, _args, *, idx=idx: self._pre_hook(idx),
            )
            post = block.register_forward_hook(
                lambda _mod, _args, _out, *, idx=idx: self._post_hook(idx),
            )
            self._hooks.extend([pre, post])

    def teardown(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._provider.cleanup()

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
