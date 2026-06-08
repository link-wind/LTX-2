# 07 FeedForward前馈网络

## 这个机制解决什么问题

读 Transformer 时，大家最容易把注意力放在 attention 上，但一个完整 block 里还有另一条非常稳定的主线：前馈网络。

它的职责不是让 token 彼此通信，而是让每个 token 在自己的通道维度上做更强的非线性变换。

## 它在整体调用链中的位置

在 `BasicAVTransformerBlock` 里，FFN 位于 self-attention 和 cross-attention 之后、block 尾部：

`x -> AdaLN -> self-attn -> cross-attn -> AdaLN -> FFN -> residual update`

视频流会走 `ff`，音频流会走 `audio_ff`。二者结构相同，只是隐藏宽度不同。

重点阅读文件：

- `packages/ltx-core/src/ltx_core/model/transformer/feed_forward.py`
- `packages/ltx-core/src/ltx_core/model/transformer/gelu_approx.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`

## 关键类与函数

- `FeedForward`
- `GELUApprox`
- `forward()`

## 数据流或张量流

它的主线非常简单：

`x -> Linear(dim -> inner_dim) + GELU(approx=tanh) -> Identity -> Linear(inner_dim -> dim_out)`

虽然 FFN 在阅读体验上常常被 attention 抢走注意力，但在真实 block 里，它承担的是每个 token 独立的通道内变换，是 attention 之外第二个最稳定的容量来源。

## 关键源码分段解析

### 1. `FeedForward` 的结构非常克制

代码没有堆很多花样，核心只有三步：

1. `GELUApprox(dim, inner_dim)`
2. `Identity()`
3. `Linear(inner_dim, dim_out)`

这里的 `Identity()` 看起来没做事，但保留顺序模块接口的统一性，后续如果要替换或插入其他层，也更方便。

### 2. `inner_dim = int(dim * mult)` 体现的是经典“先扩张再压回”

默认 `mult=4`，说明 FFN 不是在原维度上浅浅做一层变换，而是先把通道数放大，再压回输出维度。这样可以给每个 token 更多的通道内表达容量。

### 3. `GELUApprox` 把“线性投影 + 激活”合在一个小模块里

`GELUApprox` 自己内部就持有一个 `Linear`，然后调用：

`torch.nn.functional.gelu(..., approximate="tanh")`

所以从阅读视角看，`FeedForward` 其实是在用一个小封装把前半段写得更干净。

### 4. FFN 进入 block 前同样会先过 AdaLN 调制

虽然 `feed_forward.py` 文件本身很短，但真实运行时它不是裸跑的。

在 `BasicAVTransformerBlock.forward()` 里，FFN 前还会先做：

- 取出 `slice(3, 6)` 这一组 Ada 参数；
- 通过 `ada_zero_function` 做调制后的归一化；
- 再把结果送进 `ff` 或 `audio_ff`；
- 最后乘 gate 再加回 residual。

也就是说，FFN 本体很简单，但它嵌入 block 之后就进入了完整的“调制 + 残差”语境。

## 设计动机与实现取舍

为什么 FFN 这里要写得这么简单？

因为在这个项目里，复杂度主要集中在：

- 双流结构；
- 多种 attention；
- AdaLN 调制；
- backend 可插拔。

FFN 作为 block 的稳定组成部分，越简洁，越能降低整体阅读负担。

这也反过来说明一个事实：不是每个子模块都需要“花哨”。很多时候，项目的复杂性已经足够高，保留一个结构直接、行为清楚的 FFN 反而是好选择。

不要混淆两件事：

- 本篇讲的是 FFN 自身的结构和它在 block 中扮演的角色；
- [08-BasicAVTransformerBlock单层结构.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/08-BasicAVTransformerBlock单层结构.md) 才会把 FFN 连同 self-attention、cross-attention、AdaLN 一起放回单层 block 全景里。

## 一句话总结与下一篇跳转

FFN 在 LTX-2 里不是最复杂的模块，但它是 block 内稳定的第二条主干。下一篇 [08-BasicAVTransformerBlock单层结构.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/08-BasicAVTransformerBlock单层结构.md) 会把 attention、AdaLN、FFN 重新拼成一个完整的单层 block。
