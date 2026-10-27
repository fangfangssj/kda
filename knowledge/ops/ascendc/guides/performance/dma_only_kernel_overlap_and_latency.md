---
type: Guide
title: 无 VF 计算的 DMA 算子重叠与性能判读
description: 解释没有 VF 计算时 MTE/VEC 重叠指标失效的原因，并区分时延收益与双缓冲机制证据。
tags:
- task.optimization
- topic.performance.analysis
- topic.data_movement
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/performance-metrics-reference.md
  role: primary
created_at: '2026-09-14T04:03:29Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# 无 VF 计算的 DMA 算子重叠与性能判读

在来源 npusim 报告中，`pipeline_overlap` 的 VEC 操作数表示 VF 函数区间；它不是 SIMD、SIMT 之外的第四类指令。当某个实际实现只有 Scalar 地址计算和 DMA，没有 VF 工作时，MTE2/MTE3 与 VEC 的重叠自然接近零。这不能证明双缓冲未开启或不工作。gather/scatter 等算子也可能有 VF 实现，因此必须看实际代码和事件，不能按算子名判断。

此时有效分析对象是正在工作的搬运流水、Scalar 地址准备、数据粒度和总时延。来源 summary 没有直接的 MTE2↔MTE3 重叠字段；缺失不能补成零或用另一对资源的比率替代。有真实时间线时，可按忙区间并集计算所需传输重叠；只有 summary 时，机制结论应保留证据限制。

相同搬运忙时若被安排到更短 kernel 时长内，`utilization=busy/total` 会增加。因此优化后 MTE 活动比例不下降不构成失败证据。反过来，总时延下降也只证明相同工作下的效果，不能单独证明收益来自双缓冲：合批、减少冗余、改变核数或缓存状态均可能贡献收益。

确认双缓冲需要不同在途 tile 的独立槽、消费前生产完成、覆写前最后消费者完成，以及实际可重叠的搬入/搬出；仅把 bufNum 改为 2 不充分。比较时保持输入输出合同、有效字节、核配置和测量范围一致，观察总时间并尽可能关联真实 MTE2/MTE3 时间线。增加 VF 后再重新评估相应 MTE/VEC 指标，不能沿用旧实现的无计算假设。
