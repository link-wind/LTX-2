# 01 LTXModel整体结构与调用主线

## 这个机制解决什么问题

本篇先不钻进某一个算子细节，而是先回答一个更关键的问题：LTX-2 的 Transformer 主干在整个 `ltx-core` 里处在什么位置，输入是怎样进入模型的，video 和 audio 两条流又是怎样在总模型里被组织起来的。

如果这一层没有看清，后面再去读 `Attention`、`RoPE`、`AdaLN` 时，就很容易只看到局部实现，却不知道它们为什么会在那里出现。

## 它在整体调用链中的位置

可以先把主链路记成一条粗线：

`Modality -> TransformerArgsPreprocessor.prepare() -> BasicAVTransformerBlock x N -> 输出归一化/投影`

再展开一点，就是：

`video/audio Modality -> 各自的 preprocessor -> TransformerArgs -> 多层双流 block -> norm_out/audio_norm_out -> proj_out/audio_proj_out`

重点阅读文件：

- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`

## 关键类与函数

- `LTXModel`
- `LTXModelType`
- `_init_video()`
- `_init_audio()`
- `_init_audio_video()`
- `_init_preprocessors()`
- `_init_transformer_blocks()`
- `_process_transformer_blocks()`
- `_process_output()`
- `forward()`

## 数据流或张量流

这一层最重要的是先认清“输入对象”和“block 真正吃到的对象”不是同一个东西。

- 输入阶段，`LTXModel.forward()` 接收的是 `Modality`，它更像“某一模态的一包原始运行材料”，里面有 `latent`、`timesteps`、`positions`、`context`、`attention_mask`。
- 预处理阶段，`video_args_preprocessor` 和 `audio_args_preprocessor` 会把这些材料整理成 `TransformerArgs`。
- block 阶段，`BasicAVTransformerBlock` 不直接处理 `Modality`，而是处理已经补齐了时间步调制、位置编码、mask 形态的 `TransformerArgs`。
- 输出阶段，模型再把 block 的隐藏状态接到最终的 `LayerNorm + scale/shift + Linear` 输出头，生成 video/audio 的 velocity 预测。

可以把 `LTXModel` 看成一个“总装配器”：

- 它决定当前模型是 `VideoOnly`、`AudioOnly`，还是 `AudioVideo`。
- 它决定每条流该用什么宽度、多少头、什么输入/输出投影。
- 它决定是否要初始化双流之间的 cross-attention 和专用 AdaLN。
- 它负责把预处理器、block 栈和输出头串成一条完整 forward。

## 关键源码分段解析

### 1. `LTXModelType` 先决定模型到底开几条流

`LTXModelType` 不是装饰性的枚举，而是整个初始化流程的第一道分叉。

- `VideoOnly` 只初始化视频侧组件。
- `AudioOnly` 只初始化音频侧组件。
- `AudioVideo` 同时初始化两条流，并额外启用音视频 cross-attention 所需的模块。

这意味着后面看到的很多属性都不是“所有模型必有”，而是按模态条件创建的。

### 2. `_init_video()` 和 `_init_audio()` 负责各自流的局部闭环

这两个函数都在做相似的几类事：

- 输入投影：`patchify_proj` / `audio_patchify_proj`
- 时间步调制入口：`adaln_single` / `audio_adaln_single`
- 可选的 prompt AdaLN：`prompt_adaln_single` / `audio_prompt_adaln_single`
- 输出头：`scale_shift_table + norm_out + proj_out`

所以一条流并不是“只有 Transformer block”，它前后都有很明确的入口和出口。

### 3. `_init_audio_video()` 才是双流模型真正和单流模型拉开差距的地方

这里会额外创建几组专门服务于跨模态交互的 AdaLN 模块：

- `av_ca_video_scale_shift_adaln_single`
- `av_ca_audio_scale_shift_adaln_single`
- `av_ca_a2v_gate_adaln_single`
- `av_ca_v2a_gate_adaln_single`

它们对应的不是普通 self-attention，而是 audio-to-video / video-to-audio cross-attention 的缩放、平移和门控。

### 4. `_init_preprocessors()` 把“原始输入”变成“block 可消费输入”

这一步会根据模型类型选择：

- 单流模型：`TransformerArgsPreprocessor`
- 双流模型：`MultiModalTransformerArgsPreprocessor`

这里的设计很关键，因为它把“准备时间步”“准备 mask”“准备 RoPE”“准备 cross-attention timestep”这些杂事都前置掉了。于是 block 内部就能更专注在真正的注意力与残差计算上。

### 5. `_init_transformer_blocks()` 统一堆叠 `BasicAVTransformerBlock`

这一层不是把几十种不同 block 混在一起，而是把同一个 `BasicAVTransformerBlock` 重复 `num_layers` 次。

差异不是通过“第 1 层长这样、第 2 层长那样”体现的，而是通过传入的 `TransformerConfig`、模态开关、RoPE 类型、ops 配置来体现的。

### 6. `forward()` 的核心职责是“组织”，不是“发明新算子”

`LTXModel.forward()` 本身做的事情非常清楚：

1. 检查当前模型是否允许 video/audio 输入；
2. 用预处理器把 `Modality` 变成 `TransformerArgs`；
3. 调 `_process_transformer_blocks()` 让所有 block 顺序执行；
4. 调 `_process_output()` 把隐藏状态投影成最终输出。

这说明 `LTXModel` 的阅读重点不是某一个数学公式，而是“把整个运行主线串起来”的工程职责。

## 设计动机与实现取舍

这一节重点解释为什么 LTX-2 不是一个单流 Transformer，以及为什么预处理和 block 逻辑被拆到不同文件。

第一，LTX-2 明确不是单流 Transformer。

- 视频和音频的 token 结构不同。
- 视频和音频的隐藏宽度也可能不同。
- 两条流既要各自 self-attention，也要互相 cross-attention。

如果强行塞进一个统一的“单流 token 序列”，代码表面上会更短，但理解和维护都会更困难。

第二，预处理和 block 逻辑分开，是为了让职责更清晰。

- `TransformerArgsPreprocessor` 负责“准备材料”；
- `BasicAVTransformerBlock` 负责“消费材料并更新隐藏状态”；
- `LTXModel` 负责“把它们组织成完整路径”。

第三，双流实现用“同一个 block 类 + 条件创建模块”的方式，而不是搞三套完全分裂的 block。

这样做的好处是：

- 单流和双流共享大量阅读习惯；
- 同一套代码更容易保持行为一致；
- 后续替换 attention backend 或 compile 策略时，改动集中。

代价是读源码时要接受一个事实：有些属性只有特定模型类型才存在，所以阅读时必须时刻带着“当前是 video、audio，还是 audio-video”这个上下文。

## 一句话总结与下一篇跳转

如果把 `01` 看成“总地图”，那么下一篇 [02-TransformerArgs与预处理管线.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/02-TransformerArgs与预处理管线.md) 就是正式解释输入在进 block 之前被整理成了什么形状。
