---
type: Guide
title: 广播模板选路与每tile有效shape
description: 保留外层offset广播、NDDMA与UB展开的能力区别。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/advanced_tiling.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/broadcast_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/broadcast_common.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/broadcast_add_tiling.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/broadcast_add_kernel.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/datacopypad_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/nddma_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/ub_broadcast_design.md
  role: supporting
created_at: '2026-09-11T06:37:24Z'
updated_at: '2026-09-11T06:37:24Z'
platforms: []
---

# 广播模板选路与每tile有效shape

连续输入先逐轴验证广播兼容，广播轴stride0，非广播轴保留紧凑输入stride。对每输入扫描全部轴，区分是否存在广播、广播是否落在split轴及其内层、是否最后轴广播；只看第一个广播轴会遗漏混合情况。所有广播在split外层时，输入offset在广播轴不推进，每tile内部可直接连续搬；split或内层存在展开时可比较NDDMA零源stride和UB Broadcast。

UB真实形状首轴为当前rows、后续轴为对应输出/源维，源split广播时首轴1。主块ubFormer>1但尾块rows1可能退化为相同shape，应按每tile判定，无展开时合法拷贝即可。来源案例出现NaN不证明所有平台的相同shape Broadcast必然NaN；静态rank≤4、B8/B16尾轴32B对齐及BigNLast≥dcache/2约4096B是模板选择策略，需要实际API能力和测量。

NDDMA参数仅覆盖tile内rank=shapeLen-split≤5，较外维可显式循环或选其他合法算法。将负axisInsideUb夹到0只是避免数组越界，会丢维，不能支持rank>5；转UB也必须检查其rank/dtype/shape/scratch能力，不是无限fallback。源码MultiCopyConfig作const引用模板实参需适当静态存储期，普通局部constexpr并不等价。

连续布局且完整兼容广播条件下，split处输入/输出stride不同可以表示split内有复制需求，不能推广非连续布局；相同stride也需已完成广播位图验证。AddBrcTiling只用yShape[0]==1将y硬分[1,N]或[M,1]，不验证另一维，不能直接接收任意shape、标量[1,1]或双输入同时部分广播。该样例x被固定满shape，不能从多输入枚举推断完整通用实现。

UB广播紧凑源长度=(split非广播?rows:1)*内层源维积，不能按展开输出长度搬入。上游中间结果已在UB时可直接作为源，并保留producer完成与存活期；UB Broadcast不是所有可能方法中的唯一中间结果广播方式，还可有Duplicate/Gather/寄存器实现。
