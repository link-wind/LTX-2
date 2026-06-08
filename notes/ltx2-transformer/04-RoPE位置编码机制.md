# 04 RoPE位置编码机制

## 这个机制解决什么问题

注意力层只看 token 内容，本身并不知道“这个 token 在时间轴哪里、在画面哪里、在音频序列哪里”。RoPE 的职责，就是把这些位置信息变成注意力层能直接消费的旋转信号。

在 LTX-2 里，这件事比普通文本 Transformer 更复杂，因为它同时要面对：

- 视频 token 的时间/空间位置；
- 音频 token 的时间位置；
- 音视频 cross-attention 时更偏时间对齐的位置信息。

## 它在整体调用链中的位置

RoPE 的准备发生在预处理阶段，应用发生在 attention 前处理阶段：

`positions -> precompute_freqs_cis() -> positional_embeddings -> apply_rotary_emb(q, k)`

更具体一点：

- `TransformerArgsPreprocessor` 负责根据 `positions` 生成 `positional_embeddings`；
- `Attention.forward()` 里先做 `to_q/to_k`；
- `PytorchPreAttention` 再把 RoPE 应用到 `q` 和 `k` 上。

重点阅读文件：

- `packages/ltx-core/src/ltx_core/model/transformer/rope.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
- `packages/ltx-core/src/ltx_core/model/transformer/ops.py`

## 关键类与函数

- `LTXRopeType`
- `apply_rotary_emb()`
- `apply_interleaved_rotary_emb()`
- `apply_split_rotary_emb()`
- `generate_freq_grid_pytorch()`
- `generate_freq_grid_np()`
- `generate_freqs()`
- `precompute_freqs_cis()`

## 数据流或张量流

RoPE 在 LTX-2 里的路径可以简化成：

`positions -> fractional positions -> freqs -> cos/sin -> multi-head 对齐后的频率张量 -> 作用到 q/k`

这里最容易误解的一点是：RoPE 在 LTX-2 里不只是“给序列加位置”，而是要适配视频、音频、跨模态这几种不同的时间/空间组织方式。

- video 侧常见的是 3 个位置维度；
- audio 侧常见的是 1 个位置维度；
- cross-attention 会专门准备一套更偏 temporal 对齐的 cross positional embeddings。

## 关键源码分段解析

### 1. `LTXRopeType` 先决定旋转布局是 `SPLIT` 还是 `INTERLEAVED`

代码里支持两种类型：

- `INTERLEAVED`
- `SPLIT`

但注释已经提示，`INTERLEAVED` 更像 legacy 模式，当前推荐的是 `SPLIT`。这说明读 LTX-2 现代码时，优先把注意力放在 `apply_split_rotary_emb()`。

### 2. `positions` 不是简单的一维整数下标

`Modality.positions` 的默认形状是 `(B, n_pos_dims, T, 2)`，最后那个 `2` 存的是 patch 的 `[start, end)` 区间。

当 `use_middle_indices_grid=True` 时，代码会先取 patch 中点，再参与后续频率生成。这种做法比直接拿 patch 起点更贴近“这个 token 代表的真实区域中心”。

### 3. `precompute_freqs_cis()` 是频率预计算的总入口

这个函数做了几件关键事情：

1. 先生成频率网格 `indices`；
2. 再把位置网格转换为 `fractional_positions`；
3. 计算出 `freqs`；
4. 最后拆成 `cos` / `sin`，并整理成适合 multi-head attention 的形状。

也就是说，block 内部并不会现场“手算 RoPE”，而是直接消费已经准备好的 `cos/sin`。

### 4. `apply_split_rotary_emb()` 的重点不是数学公式，而是形状兼容

当然它本质上仍然是在做二维旋转，但对源码学习来说，更值得注意的是它怎么处理形状：

- 有些输入已经是 `(B, H, T, D)`；
- 有些输入还没拆成 head；
- 有些频率张量允许 batch 广播。

这也是为什么代码里会出现 `unflatten`、`transpose`、`flatten` 这些看起来有点绕的张量操作。它们的目标，是尽量在 `torch.compile` 友好的前提下完成 RoPE 应用。

### 5. cross-attention 的 RoPE 是单独准备的

`MultiModalTransformerArgsPreprocessor.prepare()` 里会额外生成 `cross_pe`。这里不是简单重用原始 `positional_embeddings`，而是专门拿 `positions[:, 0:1, :]` 这类更偏时间轴的信息，给音视频 cross-attention 做对齐。

这一步很能体现 LTX-2 的实际需求：跨模态对齐时，最重要的是时间同步，而不是把视频的完整 3D 位置原样搬过去。

## 设计动机与实现取舍

为什么 RoPE 不在 attention 里现算，而要在预处理阶段就准备好？

因为这样做能带来几个好处：

- block 内部逻辑更聚焦；
- 位置编码准备与 `positions` 的具体形态解耦；
- 单流与双流都能复用同一套预处理接口。

为什么还要保留 `SPLIT` 和 `INTERLEAVED` 两种模式？

这是一个很典型的工程妥协：

- 保留旧模式，兼容历史实现；
- 推荐新模式，统一当前主路径。

不要混淆两件事：

- 本篇关心的是“RoPE 频率是怎么准备、怎么作用到 q/k 上的”；
- [05-Attention模块总览.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/05-Attention模块总览.md) 关心的是“整个 Attention 类怎样组织 QKV、mask 和输出投影”。

## 一句话总结与下一篇跳转

RoPE 在 LTX-2 里是一条从 `positions` 延伸到 `q/k` 的完整预处理链，而不是注意力层里的一点小装饰。下一篇 [05-Attention模块总览.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/05-Attention模块总览.md) 就把 RoPE 放回真正的注意力主体里去看。
