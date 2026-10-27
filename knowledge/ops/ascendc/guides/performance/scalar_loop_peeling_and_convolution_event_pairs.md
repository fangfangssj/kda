---
type: Guide
title: Scalar 首尾剥离与卷积事件配对
description: 指出首块加载遗漏、单次重复计算及方向不配对的模板。
tags:
- task.optimization
- topic.performance.analysis
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/scalar/guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/scalar/coding_principles.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/scalar/conv_scalar.md
  role: supporting
created_at: '2026-09-11T06:42:38Z'
updated_at: '2026-09-11T06:42:38Z'
platforms: []
---

# Scalar 首尾剥离与卷积事件配对

source P2把原循环k0的SetInitialParams、周期MTE2_LoadAL1与末块SetFinalParams拆出，正例prologue却没执行k0周期加载；最后k若恰为loadStride倍数，epilogue也漏该次load。maxK1时prologue和epilogue重复Compute0，maxK0会无效Compute并可能unsigned下溢。正确剥离需把每个原迭代的副作用集合逐项映射，loadStride>0，prologue/epilogue重合时只执行一次。

Depthwise可按group→batch→M→K组织并每组一次加载权重，小case若实际本核FMAP/Weight都驻留可省部分L1重复搬运；depthwise本身不保证全部权重可放，full-load不等于不再有L0 K/M循环。group合并、尾与bias仍保留真实合同，shape特殊化必须有能力分发。

Queue/TBuf/TPipe可在已知资源布局下用LocalTensor与显式事件替代，但LocalTensor只描述视图，必须安排合法物理地址、容量、互斥和末消费者。源Conv事件先Set(M_MTE1)却Wait(MTE1_M)，随后才Set(MTE1_M)，类型不配对且首等可能无credit；这不是完整producer-consumer例。A2/L0A32768B半槽只是来源配置，字节offset和元素长度分开。

WaitFlag是会等待依赖的指令，不能称无Scalar开销；动态pingPong*常量也不必0指令。constexpr Tiling可常量传播，但图静态与直调不自动意味着编译器看到全部值，框架路径也能做部分常量化/模板分发，Host动态字段与workspace/tiling参数不可混名。改小抽象、静态字段和循环结构并非必须全部做才有收益。
