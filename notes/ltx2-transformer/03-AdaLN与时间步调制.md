# 03 AdaLN与时间步调制

## 这个机制解决什么问题

扩散模型里的 Transformer 不是在一个固定语境下反复执行同一层，而是在不同 denoising 阶段反复执行。也就是说，同一个 block 需要知道“我现在处在第几个噪声阶段”。

AdaLN 就是在解决这个问题：让时间步信息不是只在输入口轻轻加一下，而是稳定地调制到每一层计算里。

## 它在整体调用链中的位置

它既出现在预处理阶段，也出现在 block 内部：

`timesteps/sigma -> AdaLayerNormSingle -> 调制向量 -> BasicAVTransformerBlock 内各处 scale/shift/gate`

双流模型里，它还会进一步参与：

- video/audio 各自的 self-attention 前归一化
- feed-forward 前归一化
- 文本 cross-attention 的可选 AdaLN 调制
- 音视频 cross-attention 的 scale/shift 与 gate 生成

重点阅读文件：

- `packages/ltx-core/src/ltx_core/model/transformer/adaln.py`
- `packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`

## 关键类与函数

- `AdaLayerNormSingle`
- `adaln_embedding_coefficient()`
- `ADALN_NUM_BASE_PARAMS`
- `ADALN_NUM_CROSS_ATTN_PARAMS`
- `get_ada_values()`
- `get_av_ca_ada_values()`
- `apply_cross_attention_adaln()`

## 数据流或张量流

AdaLN 在这里的流向可以先粗略记成：

`raw timestep -> timestep embedding -> SiLU -> Linear -> 一大段 modulation 向量 -> 按 slice 拆成 shift / scale / gate`

这段 modulation 向量后面会被不同位置消费：

- `slice(0, 3)`：self-attention 前后的那组调制
- `slice(3, 6)`：feed-forward 前后的那组调制
- `slice(6, 9)`：可选的 text cross-attention AdaLN 调制

双流 cross-attention 还会额外走一套：

- `cross_scale_shift_timestep`
- `cross_gate_timestep`

## 关键源码分段解析

### 1. `adaln_embedding_coefficient()` 先决定一层要准备多少个调制参数

`adaln.py` 里先把这件事说得很清楚：

- 基础版本一共 6 个参数槽；
- 如果启用 `cross_attention_adaln`，再多 3 个槽。

也就是说，AdaLN 在这里不是“只输出一对 scale/shift”，而是一次性产出一整层会反复切片使用的调制池。

### 2. `AdaLayerNormSingle` 本质上是“时间步编码器 + 调制参数生成器”

它的结构并不复杂：

1. `PixArtAlphaCombinedTimestepSizeEmbeddings` 先把时间步编码成 embedding；
2. 过一个 `SiLU`；
3. 再过一个线性层，直接映射到 `embedding_coefficient * embedding_dim`。

复杂点不在层数，而在“生成出来的向量会被下游怎样切片解释”。

### 3. AdaLN 在这里不是普通 LayerNorm 的替代品

AdaLN 在这里不是“普通 LayerNorm 的替代品”这么简单。它承担的是把扩散时间步信息稳定地注入每一层计算图，让同一个 block 在不同 denoising 阶段表现出不同的调制行为。

换句话说，这里真正重要的不是“归一化”三个字，而是“自适应调制”。

### 4. `get_ada_values()` 负责把大向量拆成 block 当前要用的小块

在 `BasicAVTransformerBlock` 里，`scale_shift_table` 更像是一张可学习底表，`timestep` 更像运行时条件。两者相加后，再按 `slice` 切成：

- `shift`
- `scale`
- `gate`

这里的设计非常实用：同一层所有 AdaLN 相关参数都来自同一份连续表示，但在使用时能针对不同子层拆开。

### 5. `PytorchAdaZeroFunction` 是真正把 scale/shift 用到隐藏状态上的地方

在 `ops.py` 里，它的核心逻辑很直接：

`rms_norm(x) * (1 + scale) + shift`

所以流程不是“先做一个普通 RMSNorm，再随便加个条件”，而是让时间步直接介入归一化后的重缩放和重平移。

### 6. 双流 cross-attention 还有一套专门的 AdaLN 通路

`get_av_ca_ada_values()` 和 `_prepare_cross_attention_timestep()` 说明了一件很重要的事：音视频互相看见对方时，调制信息不只来自“我自己的 timestep”，还会来自“另一模态当前的 sigma”。

这也是双流扩散模型比单流更难读的地方之一：不仅有每条流自己的时间步，还有跨模态交互时额外生成的 gate 与 scale/shift。

## 设计动机与实现取舍

为什么这里要用 AdaLN，而不是把时间步直接加到 token 上？

因为“加到输入上”只能提供一次、较浅的条件注入；而 AdaLN 可以让条件深入到每层的归一化和残差更新里。对于扩散模型来说，这通常更稳定，也更有表达力。

另外，这里采用的是“先生成大向量，再按用途切片”的方案。

优点：

- 每层需要的调制值来源统一；
- 写 block 时不需要维护很多分散的小头；
- 加 cross-attention AdaLN 时，只是扩展槽位，而不是推翻整套接口。

代价：

- 初读时不容易看出 `slice(0, 3)`、`slice(3, 6)`、`slice(6, 9)` 分别代表什么；
- 读者需要同时理解 `scale_shift_table`、`timestep` 和 `gate` 三者的关系。

## 一句话总结与下一篇跳转

这一篇要记住的核心是：AdaLN 让时间步从“输入条件”变成“层级调制信号”。下一篇 [04-RoPE位置编码机制.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/04-RoPE位置编码机制.md) 会接着看另一条同样重要的条件线索：位置信息是怎样通过 RoPE 进入注意力计算的。
