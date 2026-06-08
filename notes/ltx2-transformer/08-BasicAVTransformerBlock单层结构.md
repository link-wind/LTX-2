# 08 BasicAVTransformerBlock单层结构

## 这个机制解决什么问题

前面几篇已经把预处理、AdaLN、RoPE、Attention、FFN 分开看过了，但真实运行时它们不是分散存在的，而是都被装进一个单层 block 里反复堆叠。

`BasicAVTransformerBlock` 就是这个最关键的学习单位。读懂一层，后面 48 层基本就是同一种模式在重复。

## 它在整体调用链中的位置

它位于 `TransformerArgs` 和最终输出头之间：

`TransformerArgs -> BasicAVTransformerBlock x N -> output norm/proj`

如果只看一层，主线大致是：

`self-attn -> text cross-attn -> optional audio-video cross-attn -> FFN`

重点阅读文件：

- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`
- `packages/ltx-core/src/ltx_core/model/transformer/ops.py`

## 关键类与函数

- `BasicAVTransformerBlock`
- `get_ada_values()`
- `get_av_ca_ada_values()`
- `_apply_text_cross_attention()`
- `forward()`
- `apply_cross_attention_adaln()`

## 数据流或张量流

如果只看 video 流，一层内部的大致流向是：

`vx -> AdaLN(self) -> self-attn -> residual -> text cross-attn -> residual -> AdaLN(ffn) -> FFN -> residual`

audio 流同理，只是维度与对应模块换成音频侧版本。

双流时还会插入：

`audio -> video cross-attn`

和

`video -> audio cross-attn`

这一步会把单层 block 从“两个各算各的分支”升级成“两个分支之间还会互相交换信息”的结构。

## 关键源码分段解析

### 1. 初始化阶段先决定这一层到底包含哪些子模块

`BasicAVTransformerBlock.__init__()` 不是无脑创建所有组件，而是按 `video` / `audio` 配置条件化地创建：

- 视频侧 `attn1`、`attn2`、`ff`
- 音频侧 `audio_attn1`、`audio_attn2`、`audio_ff`
- 双流交互时的 `audio_to_video_attn` 与 `video_to_audio_attn`

所以这个 block 其实是一个“能覆盖单流和双流”的通用骨架。

### 2. `get_ada_values()` 让一层里的多处调制都来自同一张底表

`scale_shift_table + timestep` 先合成一份运行时参数，再按 slice 拆成当前子层需要的 `shift / scale / gate`。

这一步的结果会被后面 self-attention、FFN、甚至可选 cross-attention AdaLN 多次消费。

### 3. self-attention 是每条流最先执行的局部建模步骤

以 video 为例，主线是：

1. 先从 `slice(0, 3)` 拿到 self-attention 相关 Ada 参数；
2. 用 `ada_zero_function` 做调制后的归一化；
3. 把结果送进 `attn1`；
4. 用 `gate` 控制这次 attention 输出有多大；
5. 残差加回原 `vx`。

audio 侧是同一模式的镜像版本。

### 4. 文本 cross-attention 紧跟在 self-attention 后面

`_apply_text_cross_attention()` 做了一个很好的职责收口：

- 如果没开 `cross_attention_adaln`，就直接做常规 cross-attention；
- 如果开了，就额外给 query 和 prompt hidden states 做 AdaLN 调制。

这让 `forward()` 主干保持相对简洁，而不必把文本 cross-attention 的分支逻辑全摊在主流程里。

### 5. 音视频 cross-attention 是单层 block 最特别的部分

这一段最容易让读者迷路，但也最能体现 LTX-2 的独特性。

- `run_a2v`：audio 提供上下文，video 去看 audio；
- `run_v2a`：video 提供上下文，audio 去看 video。

而且代码里还特意先保存 `vx_pre_av` 和 `ax_pre_av`，避免先执行一个方向后污染另一个方向的键值来源。这是很细但很重要的工程处理。

### 6. FFN 作为单层 block 的收尾

在跨模态交互之后，每条流还会再过一轮：

- `slice(3, 6)` 对应的 AdaLN 调制；
- `ff` / `audio_ff`
- gate 控制后的残差更新

这样一层 block 才真正闭环。

## 设计动机与实现取舍

`08` 的重点是“单层 block 内部怎么拼起来”，不是完整 forward。

这一层的设计非常强调复用：

- 同一个 block 类覆盖单流和双流；
- 同一套 AdaLN 取值逻辑复用在多个子层；
- 同一个 Attention 壳复用在 self、text cross、audio-video cross 几种场景。

这样做让代码更统一，但也要求阅读者始终记住：同一个函数体里，可能同时包含“只在 video 开启时跑”的分支、“只在 audio 开启时跑”的分支，以及“双流都存在时才跑”的分支。

因此学习时最稳的方式不是逐行死抠，而是先抓住单层的固定顺序，再分别把 video/audio/double-stream 的条件分叉补进去。

## 一句话总结与下一篇跳转

`BasicAVTransformerBlock` 是把所有核心机制装配成真实一层的地方。下一篇 [09-音视频双流交互机制.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/09-音视频双流交互机制.md) 会单独把其中最特别、也最容易绕的音视频双流交互拿出来看。
