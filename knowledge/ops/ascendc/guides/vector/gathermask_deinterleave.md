---
type: Guide
title: GatherMask 解交织的源跨度与输出长度
description: 在 A2/A3 参考范围内区分固定模式、自定义选择掩码、repeat 源跨度和压缩后输出长度。
tags:
- task.optimization
- topic.compute.indexing
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-gathermask.md
  role: primary
created_at: '2026-09-10T12:49:29Z'
updated_at: '2026-09-10T12:49:29Z'
platforms:
- a2
- a3
---

# GatherMask 解交织的源跨度与输出长度

交织数据按固定相位提取时，GatherMask 可以将被选中的元素收紧到输出。来源在 A2/A3 范围内给出固定 pattern：1/2 分别选择偶数/奇数索引，3–6 分别选择每四个元素中的一个位置；更复杂的选择可使用受支持类型的 tensor mask。具体 dtype、参数范围和模式仍应与安装版本匹配。

源元素数与提取后的输出元素数必须分别计算。例如来源的 FP32 Normal 模式中，一个 256 字节源 repeat 包含 64 个元素，“每四取一”输出 16 个元素。连续 repeat 的源起点应推进整个源跨度；将 32 字节 block 步长误当成 256 字节 repeat 步长，会重复读取上一批的大部分数据。

固定 pattern 不需要每轮读取自定义 mask tensor；自定义模式还需核对 mask 元素类型及 mask 的重复步长。Normal 和 Counter 模式的数据量控制不同，不能复用同一组未经解释的 repeat/count 参数。混用模式时还应核对接口调用后的 mask 状态。

满 repeat 和尾部应分开描述：源覆盖不得越界，输出只包含有效选择结果。尾部回退、缓冲复用和后续计算的同步都需保持真实依赖。来源中的 tile≥2048、固定 UB 容量及固定循环顺序属于场景经验，不能作为所有算子的硬约束。

复用解交织结果可以减少重复搬运，但会延长缓冲生命周期。性能对照应包括解交织、搬运和消费者的总成本；正确性应覆盖每种相位、不满 repeat、多个 repeat 与自定义 mask。
