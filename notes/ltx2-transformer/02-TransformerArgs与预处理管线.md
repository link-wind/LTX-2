# 02 TransformerArgs与预处理管线

## 这个机制解决什么问题

`LTXModel` 接收到的输入还是比较“原始”的 `Modality`；而 `BasicAVTransformerBlock` 想吃到的，却是一份已经准备好时间步、位置编码、文本条件和 mask 的结构化输入。

`TransformerArgs` 和它背后的预处理器，就是专门解决这个落差的。

本篇的目标是回答：为什么模型不把 `Modality` 直接塞进 block，而要先走一层 `prepare()`。

## 它在整体调用链中的位置

它正好卡在“输入对象”和“真正的 Transformer 计算”之间：

`Modality -> TransformerArgsPreprocessor.prepare() -> TransformerArgs -> BasicAVTransformerBlock`

双流时则是：

`video/audio Modality -> MultiModalTransformerArgsPreprocessor.prepare() -> 带 cross 信息的 TransformerArgs -> 双流 block`

重点阅读文件：

- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
- `packages/ltx-core/src/ltx_core/model/transformer/modality.py`
- `packages/ltx-core/src/ltx_core/model/transformer/model.py`

## 关键类与函数

- `Modality`
- `TransformerArgs`
- `BlockPerturbationsProcessor`
- `TransformerArgsPreprocessor`
- `MultiModalTransformerArgsPreprocessor`
- `_prepare_timestep()`
- `_prepare_context()`
- `_prepare_attention_mask()`
- `_prepare_self_attention_mask()`
- `_prepare_positional_embeddings()`

## 数据流或张量流

这里最值得记住的是：`TransformerArgs` 不是“多包了一层壳”，而是明确把 block 运行所需的字段一次性备齐了。

`Modality` 里的关键输入包括：

- `latent`
- `sigma`
- `timesteps`
- `positions`
- `context`
- `context_mask`
- `attention_mask`

进入 `prepare()` 之后，会被整理成：

- `x`：已经做完输入线性投影的隐藏状态
- `context`：已经对齐到 block 隐藏维度的文本条件
- `timesteps`：后续 AdaLN 会直接消费的调制向量
- `embedded_timestep`：输出头还会继续用到的时间步嵌入
- `positional_embeddings`：供 self-attention 使用的 RoPE 频率
- `self_attention_mask`：已经整理成 attention 后端易用形态的 mask
- 双流附加字段：`cross_positional_embeddings`、`cross_scale_shift_timestep`、`cross_gate_timestep`

可以把它理解成一张“block 输入清单”。

## 关键源码分段解析

### 1. `Modality` 是运行时输入的原始契约

`Modality` 的职责是把某个模态当前 forward 需要的材料捆在一起。

这里有几个点特别值得注意：

- `latent` 已经是 patch/token 级别，而不是原始像素或波形；
- `positions` 对 video 默认是 3 维位置，对 audio 默认是 1 维位置；
- `attention_mask` 是比较高层、语义化的 `[0, 1]` 掩码，不是最终 backend 直接能吃的 bias。

所以 `Modality` 的设计更偏“业务输入结构”，而不是“底层算子输入结构”。

### 2. `TransformerArgs` 是 block 视角下的标准化输入

`TransformerArgs` 用 dataclass 把 block 需要的字段固定下来。它的价值在于：

- block 不需要再去猜某个字段要不要算；
- 预处理完成后，所有层都能复用同一种输入契约；
- 后面叠加 perturbation mask 时，也可以直接通过 `replace()` 产生新的同构对象。

这正是工程上常见的一个好设计：把“前处理阶段的杂乱输入”压成“计算阶段的稳定接口”。

### 3. `patchify_proj`、timestep、mask、RoPE 都在预处理阶段落地

这一段最容易帮助初学者建立直觉：

- `patchify_proj` 把原始 latent 映射到 block 需要的隐藏维度；
- `adaln` 先把时间步变成后续调制会用到的向量；
- `context_mask` 和 `self_attention_mask` 会在进入 attention 前被整理成可广播的形态；
- `positional_embeddings` 不是在 block 里临时现算，而是在预处理阶段就准备好。

这样 block 内部就不必重复做这些准备动作，也更不容易把“准备逻辑”和“核心计算逻辑”混在一起。

### 4. `_prepare_timestep()` 同时产出两类时间步表示

这个函数返回的是两个值：

- 调制向量 `timestep`
- 原始嵌入 `embedded_timestep`

前者主要服务于 block 里的 AdaLN 调制，后者则还要留给输出头继续使用。也就是说，时间步信息不是“一次用完”，而是贯穿了 block 内外两段路径。

### 5. `_prepare_attention_mask()` 和 `_prepare_self_attention_mask()` 分别处理两类 mask

这两个函数容易混。

- `context_mask` 主要给文本 cross-attention 用；
- `self_attention_mask` 主要给模态内部 self-attention 用。

尤其 `_prepare_self_attention_mask()` 这一步很重要：它不是简单转 dtype，而是把 `[0, 1]` 形式的掩码变成 attention backend 更容易消费的加性 log-space bias。

### 6. `MultiModalTransformerArgsPreprocessor` 在单流预处理之上再加一层跨模态准备

双流模式下，并不是重写一整套逻辑，而是：

1. 先复用 `TransformerArgsPreprocessor` 做基础准备；
2. 再补上跨模态所需的时间步和 cross RoPE。

这是一种很干净的扩展方式：先有单流基类能力，再在双流场景里增量添加跨模态字段。

## 设计动机与实现取舍

为什么要先有 `Modality`，再转成 `TransformerArgs`？

因为这两个对象面向的问题本来就不同。

- `Modality` 解决的是“调用者应该提交什么”；
- `TransformerArgs` 解决的是“block 真正需要什么”。

如果把这两个角色混成一个 dataclass，表面上会少一个类型，实际上会让输入契约和内部契约互相污染。

另外，这里还有一个很重要的工程收益：预处理器让 block 变得更“瘦”。

- block 不负责构造 RoPE；
- block 不负责把 mask 从业务格式转成算子格式；
- block 不负责理解 prompt 投影或 timestep 嵌入生成。

这样一来，`BasicAVTransformerBlock` 可以更专心地表达“这一层怎么算”，而不是“算之前要准备什么”。

## 一句话总结与下一篇跳转

这一篇的核心结论是：`TransformerArgs` 是 block 世界里的标准输入格式，预处理器负责把原始 `Modality` 清洗成这个格式。下一篇 [03-AdaLN与时间步调制.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/03-AdaLN与时间步调制.md) 会继续解释其中最关键的一块：时间步是怎样被注入每一层计算的。
