---
type: Guide
title: SIMD VF 指标的分母、发射停顿与代码形态
description: 区分 SIMD VF 的调用数、工作量、发射间隙和累计时长，约束代码形态诊断所依赖的统计范围。
tags:
- task.optimization
- topic.performance.analysis
- topic.programming_model.execution_model.simd
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/performance-metrics-reference.md
  role: primary
created_at: '2026-09-14T04:03:29Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# SIMD VF 指标的分母、发射停顿与代码形态

npusim 来源中的 `simd_vf_metrics` 描述 VF 调用层面的代码形态。`vf_distinct_count` 是调用点数，`vf_total_executions` 是累计调用次数，`vf_active_cores` 是本份数据中出现 VF 的 AIV 子核数，`vf_executions_per_core` 是二者相除。比较形状或核数变化时优先核对分子与分母来自相同采集集合；归一化不能恢复未采集核的真实分布。

`vf_total_cycles` 累加各 VF 的时长，可能跨核并行而重叠，不能直接视为 kernel 墙钟时间。avg/min/max 描述不同调用样本；合并多个报告时不能再不加权平均这些平均值。`rvecsu_cycles/vf_total_cycles` 和 `pushq_cycles/vf_total_cycles` 表达来源定义的开销比，但若分子跨调用/核重复统计，则比值大于 1 不应被解释为物理时间百分比。先核验生产者范围，再讨论参数准备和调用碎片。

来源把 VF stall tick 定义为某次 VF 内没有指定向量类指令开始执行的 tick；调用间主 Scalar 准备参数的时间不在该 VF 窗口内。`vf_total_stall_cycles` 是这些 tick 的累计数，`vf_stall_event_count` 是连续无发射段的段数，`vf_stall_ratio=stall_ticks/vf_total_cycles`。一个持续 19 ticks 的无发射段与 19 个单 tick 段具有相同累计时长、不同段数。没有新指令发射不等于已有指令完成或硬件全部空闲。

调用频繁、每次工作少可提示融合或扩大每次 VF 工作；单个累加器形成的依赖链可提示独立累加器和最后合并。但 `rv_vst/(rv_vadd+rv_vmax)`、Reduce 指令计数、ExecIPC、每调用 stall 段数等只提供定位线索，不能替代读取数据流。共享 SIMD/SIMT 指令计数与去重后的 VF 调用数混用，还可能放大 `rvecex_per_vf`。

0.1/0.2 stall 比、每核 8 次调用、0.15 Scalar 开销等均为来源启发式，不能据一次阈值触发断言根因，或把低于阈值的短依赖链判为不存在。优化需验证真实数组布局、累加顺序、寄存器压力、尾部和数值合同，再比较相同工作下的 kernel 时长、VF 工作量及调用数量。阈值下降本身不等同于总性能提高。
