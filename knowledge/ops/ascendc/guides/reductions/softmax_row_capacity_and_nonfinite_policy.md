---
type: Guide
title: 整行 Softmax 的容量与非有限值策略
description: 保存逐行稳定计算和BF16流水，区分逻辑列数与物理行距。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/softmax/softmax_tiling_example_for_lowering.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/softmax/softmax_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/softmax/softmax_process_example_for_lowering.md
  role: supporting
created_at: '2026-09-11T05:05:12Z'
updated_at: '2026-09-11T05:05:12Z'
platforms: []
---

# 整行 Softmax 的容量与非有限值策略

该示例为二维[B,N]紧密行布局、每行完整归约。行数按pivot分配，各核从rowStart*N读取myRows行；不是任意axis或任意stride的通用Softmax。m=max(x)，e=exp(x-m)，s=sum(e)，y=e/s对有限非空行提供稳定计算，来源用Muls(e,1.0f/s)仍包含Scalar倒数舍入，不能因避开Vector Reciprocal就保证精确。

源BF16搬入后升FP32，ReduceMax/ReduceSum共用sharedBuf，再降BF16输出；src/dst/scratch共享须满足实际重载。GetValue读取统计量与后续Vector使用负max/倒数都有相应完成条件。正负Inf、NaN、全-inf行或N0不能由减max公式无条件处理，inf-inf及-inf-(-inf)会NaN，按算子合同定义或保留未支持范围。

两份BF16 I/O加四份FP32(input/output/exp/shared)为20N有效字节，另需对齐和scratch真实上限。Host直接tileLength=nCols而无容量检查，不保证整行适配任何N。大行可用分块多遍或在线统计等合法算法，不能通过截断列数装入UB。

普通DataCopy BF16 count=N需要对应长度约束；N非16倍数时不能照抄整行路径。物理UB行距和容量按padding，统计count仍是逻辑N，max padding需中性值且sum不能混入无效元素。nRows0导致usedCoreNum0和除零，任意rank读取dim1也需守卫；ND注册不自动证明二维紧密存储。

完整Host/Init/Process各自重复的字段要保持同版；只编译BF16注册案例不能证明模板支持float/half。源码中未实现的阶段是教学中间产物，不能作为已经通过边界测试的完整实现。
