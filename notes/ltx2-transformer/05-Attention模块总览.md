# 05 Attention模块总览

## 这个机制解决什么问题

无论是 video self-attention、audio self-attention，还是文本 cross-attention、音视频 cross-attention，最后都要落到同一个核心问题上：给定查询 `q`、键 `k`、值 `v`，这一层怎样算出新的隐藏状态。

LTX-2 把这件事集中封装在 `Attention` 类里。读懂它，就等于读懂了 block 里最核心的子层之一。

## 它在整体调用链中的位置

在 block 内部，Attention 大致处在这样的链路里：

`x -> to_q/to_k/to_v -> preattention(q,k,pe) -> attention backend -> 可选 gating -> to_out`

如果放回 `BasicAVTransformerBlock` 里，它会被用于：

- `attn1`：模态内 self-attention
- `attn2`：文本 cross-attention
- `audio_to_video_attn`：audio 作为上下文供 video 查询
- `video_to_audio_attn`：video 作为上下文供 audio 查询

重点阅读文件：

- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`
- `packages/ltx-core/src/ltx_core/model/transformer/ops.py`

## 关键类与函数

- `Attention`
- `AttentionOps`
- `AttentionCallable`
- `MaskedAttentionCallable`
- `PytorchPreAttention`
- `PytorchGatedAttention`
- `forward()`

## 数据流或张量流

先抓住最稳定的一条主线：

`x -> q`

`context -> k, v`

然后经过：

- `q_norm / k_norm`
- 可选 `RoPE`
- masked 或 unmasked attention backend
- 可选 `perturbation_mask`
- 可选 per-head gating
- `to_out`

如果 `context is None`，那就是 self-attention；如果 `context` 单独传入，那就是 cross-attention。

## 关键源码分段解析

### 1. `Attention.__init__()` 把“结构参数”和“算子实现”分开

初始化时，既会创建结构上的线性层和归一化层：

- `to_q`
- `to_k`
- `to_v`
- `q_norm`
- `k_norm`
- `to_out`

也会从 `AttentionOps` 里拿到底层调用策略：

- `attention_function`
- `masked_attention_function`
- `preattention_function`
- `gated_attention_function`

这说明 `Attention` 本身既是一个模块壳，也是若干可插拔算子的调度点。

### 2. `context = x if context is None else context` 是 self/cross 的分界线

这是一个很经典但很重要的设计：

- 不传 `context`，`k/v` 就和 `q` 来自同一份输入，形成 self-attention；
- 传入 `context`，`q` 和 `k/v` 的来源分开，形成 cross-attention。

因此同一个 `Attention` 类可以复用到多种注意力场景，不必为 self 和 cross 分别写两套模块。

### 3. `PytorchPreAttention` 负责 Attention 前处理

真正进入 backend 前，还要做两件事：

- `q_norm` 和 `k_norm`
- 如果有 `pe`，对 `q` 和 `k` 应用 RoPE

这也是为什么前一篇单独讲 RoPE 是值得的：它不是独立于 Attention 的另一层，而是 attention 前处理链的一部分。

### 4. 有 mask 和没 mask 会走不同 backend 接口

`Attention.forward()` 里有个非常清楚的分叉：

- `mask is None`：走 `attention_function`
- `mask is not None`：走 `masked_attention_function`

这不是代码风格问题，而是因为并不是所有高性能 attention backend 都支持 mask。

### 5. `perturbation_mask` 是 LTX-2 里很有工程味道的一层旁路

这里如果 `perturbation_mask` 存在，输出会做：

`out = out * perturbation_mask + v * (1 - perturbation_mask)`

也就是说，某些情况下模型可以让注意力输出和原始 value 投影做混合，而不是非黑即白地“全部算 attention”或“全部跳过 attention”。

### 6. 可选 gating 是 attention 输出后的再调制

如果 `apply_gated_attention=True`，模块会额外创建 `to_gate_logits`。之后 `PytorchGatedAttention` 会把输出 reshape 成 `(B, T, H, D)`，按 head 维度施加门控，再 reshape 回去。

这意味着 gating 不是替代 attention，而是注意力已经算完之后的逐头重加权。

## 设计动机与实现取舍

LTX-2 的 `Attention` 很值得学习的一点，是它没有把所有逻辑硬塞进一个巨大 `forward`。

- 线性投影归模块自己管；
- preattention 归 `PytorchPreAttention` 这类 callable 管；
- backend 选择归 `AttentionOps` 管；
- gating 归单独 callable 管。

这样做的好处是模块边界清晰，可替换性强。

代价是初读时要同时追几层抽象，不像最简单的手写 attention 那样一屏就能看完。

不要混淆两件事：

- 本篇讲的是 Attention 类本身怎么接输入、拆 QKV、走 self/cross 两种路径；
- [06-Attention后端与算子选择.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/06-Attention后端与算子选择.md) 讲的是同一套 attention 接口背后，为什么还能切到 SDPA、xFormers 或 FlashAttention。

## 一句话总结与下一篇跳转

这一篇要抓住的是：`Attention` 是一个统一壳，内部再把 RoPE、mask、backend、gating 串起来。下一篇 [06-Attention后端与算子选择.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/06-Attention后端与算子选择.md) 专门看这个“统一壳”下面为什么还能换不同底层实现。
