# 12 LTX-2 Transformer学习地图

## 这个机制解决什么问题

前 11 篇已经把主干代码拆成若干主题，但学到最后很容易出现另一种问题：每一篇都懂一点，合在一起反而不知道该怎么复习、该先重读哪几篇、哪些概念最容易串线。

这一篇的目标，就是把前面的内容压缩成一张学习地图。

## 它在整体调用链中的位置

它不对应新的运行模块，而对应“如何回看整个 Transformer 目录”。

从学习顺序上，它位于全系列的最后；从复习顺序上，它反而应该被反复拿出来用。

重点回看文档：

- [01-LTXModel整体结构与调用主线.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/01-LTXModel整体结构与调用主线.md)
- [02-TransformerArgs与预处理管线.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/02-TransformerArgs与预处理管线.md)
- [08-BasicAVTransformerBlock单层结构.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/08-BasicAVTransformerBlock单层结构.md)
- [10-模型前向传播全链路梳理.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/10-模型前向传播全链路梳理.md)

## 关键类与函数

这一篇不引入新类，而是给前面出现频率最高的对象做复习定位：

- `LTXModel`
- `Modality`
- `TransformerArgs`
- `TransformerArgsPreprocessor`
- `BasicAVTransformerBlock`
- `Attention`
- `AdaLayerNormSingle`
- `TransformerConfig`
- `TransformerOpsConfig`

## 数据流或张量流

如果只保留一条最重要的总链路，可以记成：

`Modality -> TransformerArgsPreprocessor.prepare() -> TransformerArgs -> BasicAVTransformerBlock x N -> output head`

如果要再细一层，就把 block 内部记成：

`AdaLN -> self-attn -> text cross-attn -> optional audio-video cross-attn -> FFN`

如果要再补一个“配置层”的视角，就加上：

`结构参数 -> block 初始化`

`执行参数 -> attention backend / compile 策略`

## 关键源码分段解析

### 1. 第一遍重读应该先抓入口，不要先钻优化细节

建议第一遍复习时把注意力放在：

- `model.py`
- `transformer_args.py`
- `transformer.py`

先确认“输入如何进来、block 如何执行、输出如何出去”，再去补 `attention.py`、`rope.py`、`compiling.py` 这类更细的实现。

### 2. 推荐阅读顺序就是这组文档的骨架

1. 先读 `01-LTXModel整体结构与调用主线.md`
2. 再读 `02-TransformerArgs与预处理管线.md`
3. 接着读 `03` 到 `07` 这些机制篇
4. 然后读 `08` 到 `10`，把 block 和全链路拼起来
5. 最后用 `11` 和 `12` 做回顾与扩展

这个顺序的核心逻辑是：先建立地图，再学习零件，最后回到总装。

### 3. 最容易混淆的三个点

第一，`Modality` 和 `TransformerArgs` 不是一回事。

- 前者面向输入契约；
- 后者面向 block 内部契约。

第二，AdaLN 和普通归一化不是一回事。

- 这里的重点是时间步调制；
- 不是单纯把 `LayerNorm` 换成另一种 norm。

第三，双流模型不是“两个单流模型并排放着”。

- 它们之间还会通过 cross-attention 交换信息；
- 而且这种交互有专门的 RoPE、AdaLN 和 gate。

### 4. 如果你只剩很短时间，优先回看哪几篇

如果是快速回顾，最推荐回看：

- `01`：建立总图
- `02`：看懂输入怎样变成 block 可消费格式
- `08`：看懂单层 block 真正如何拼装
- `10`：看懂完整 forward

如果是准备继续深挖优化层，再补：

- `06`：attention backend
- `11`：可插拔与 compile 设计

## 设计动机与实现取舍

这组笔记本身也体现了一种取舍：不用“按文件硬讲”的方式，而是按阅读心智负担更低的方式，把源码拆成“入口、预处理、调制、位置、注意力、单层、双流、全链路、配置”这几类主题。

这样做的好处是更适合学习和复盘；代价是某些真实文件会在多篇文档里反复出现。但对于源码学习来说，这种“按问题组织知识”通常比“按文件组织知识”更有效。

如果你后面继续往仓库外扩展，我建议下一站优先看：

- `packages/ltx-core/src/ltx_core/components/patchifiers.py`
- `packages/ltx-core/src/ltx_core/text_encoders/`
- `packages/ltx-pipelines/src/ltx_pipelines/`

这样就能把“Transformer 主干”继续接到更完整的推理路径上。

## 一句话总结与下一篇跳转

这篇没有下一篇要跳了。最重要的结论是：把 `LTXModel -> TransformerArgs -> Block -> 输出头` 这条总线记住，再回头看任何局部机制，都会轻松很多。
