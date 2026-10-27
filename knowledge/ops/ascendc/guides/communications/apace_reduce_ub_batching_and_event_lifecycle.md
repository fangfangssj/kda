---
type: Guide
title: APACE 批量归约的 UB 与事件生命周期
description: APACE 批量归约的 UB 与事件生命周期的来源契约、适用范围和失效边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/fusion.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/communication.md
  role: supporting
created_at: '2026-09-11T07:14:32Z'
updated_at: '2026-09-11T07:14:32Z'
platforms: []
---

# APACE 批量归约的 UB 与事件生命周期

每轮输出行区间不相交，收齐该轮各rank贡献后即可归约，数学上不需跨轮acc状态。典型BF16输入用独立FP32 cast后累加，最后CAST_RINT回BF16；结果仍受累加顺序、输入量化及最终舍入影响。仅在暴露于总时延且全量精度成立时才有降精度收益，不能从减少Cast数量推定端到端加速。

六块布局为2份BF16 src、2份FP32 cast、1份FP32 acc、1份BF16 out，共18字节/物理元素；物理行pitch、对齐、comm/barrier区、guards均另计。180KiB/18=10240元素上限是紧凑理论，7680元素需138240B，不能称180KiB精确可装上限。N过大分列段，行批次按pitch预算。32B/64B行距取决于具体搬运路径，不能把性能示例64B变所有dtype硬约束。

2D批量搬运可将每行同步摊为每批同步，事件数量约ceil(rows/batchRows)*Nsegments*R。TPipe/TQue分配的LocalTensor也可表达多行并使用2D DMA；“只有手动UB能批量”是错误限制。静态地址与TPipe可以在物理隔离且生命周期闭合的布局共存，guard需覆盖真实通信区域，不是见两种API就必错。

MTE2_V保证搬入后Cast可读，V_MTE2保证V消费后同src slot可重装，V_MTE3保证输出转换后可搬出，MTE3_V保证搬出后可重写out。pingpong跨迭代Set/Wait是合法依赖：i>=2时等同slot旧消费者；起始credit、R=1和结束残留消费必须闭合。文档“所有同迭代配对”与它自己的跨迭代模板冲突，不能照抄。

源特定strided多行路径报告29–32行丢数据，需以版本、描述符范围和复现证据界定；可分批或在该分支逐行DMA但整批管理事件，不将32当所有硬件统一上限。单核归约只要唯一写者可正确，是否性能不足由工作量决定。宽化Cast别名必须遵循具体API允许范围，独立buffer易证明安全，不泛化“任何原地一定数学错误”。
