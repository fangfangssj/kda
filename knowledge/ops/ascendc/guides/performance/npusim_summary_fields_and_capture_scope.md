---
type: Guide
title: npusim summary 字段、采样范围与缺失值
description: 解释 npusim summary 的时间、核范围、流水活动和可选计数，避免把缺失或采样子集当作硬件事实。
tags:
- task.optimization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/performance-metrics-reference.md
  role: primary
created_at: '2026-09-14T04:03:29Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# npusim summary 字段、采样范围与缺失值

固定来源的 `summary.json` 用 `kernel_info.kernel_total_clocks` 表达仿真 kernel 的整体 tick 时长，`kernel_instructions_executed` 表达所统计指令总数。两者用途不同：减少指令可能降低冗余，但也可能只是改变被采集的工作；展开或并行化可能增加指令却缩短时间。只有同输入、同有效输出、同采集范围下的时长才可比较。

`ai_core_active` 的文字说明同时涉及可分析核数和 launch 核数，不能独立充当真实活跃物理核数证明。该值为 1 既可能来自单核执行，也可能来自只采一核。`per_core` 长度小于报告核数时，均值、极值、`imbalance_ratio` 的计算人口必须回到生产者核对；不能凭数组位置识别真实慢核，也不能把缺少其他核的观测补成均衡或零活动。

`pipe_utilization.pipeline_util_summary` 以 `{SUBCORE}_{PIPE}` 区分 AIC/AIV 及 SCALAR、CUBE、SIMD、SIMT、MTE 等资源，字段通常包含 mean/min/max 和可选 per_core。来源把活动比例的分母写为 kernel_total_clocks；消费其他版本时必须确认实际时间窗口。SIMT 与 SIMD 是独立的执行侧，SIMD 接近零不代表 Vector 子核没有 SIMT 工作。RVECEX/RVECLD/RVECST 在该报告中可能聚合 SIMD 与 SIMT，不能当成纯 SIMD 指令数。

`top_level_diagnosis.dominant_pipeline` 是所比较活动均值的最大项，不自动等同于临界路径或可优化根因。来源 `imbalance_ratio=max_core_duration/min_core_duration` 需完整核集合、有效分母和一致时间口径；1.3/2.0 等阈值只是分析启发式。真实工作较少、启动成本、等待和缺采均可能改变比例。

`cache`、`bandwidth` 可缺失，字段 null 表示不可用时不得转为 0。`simd_vf_metrics` 的空对象与缺字段应按 schema 记录；来源对其必现/可缺表述存在差异。cache refill 不独立证明寄存器溢出，需指令、地址和编译后的 spill 证据。带宽字段 `total_gbps`、单事务速率和峰值比例有不同分母；HBM 产品峰值不能直接套到 L1/L0/UB 传输或单核采样。单笔事务仍是一个观察，只是不能据此推断稳定吞吐。

逐核统计应保存可观测核映射、有效窗口、单位、空值和采集版本。核间局部性优化在仿真中收益较小不证明真机一定收益更大；真机效果需独立验证。
