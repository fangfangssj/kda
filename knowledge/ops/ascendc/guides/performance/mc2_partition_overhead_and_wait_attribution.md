---
type: Guide
title: MC2 切分开销与通信等待归因
description: 联合考察通信粒度、Matmul 利用率与依赖等待，避免仅按切分数或 MTE2 占比推断 MC2 性能瓶颈。
tags:
- task.optimization
- topic.performance.analysis
- topic.data_movement
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/comm-compute/expansion_analysis.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/comm-compute/bound_diagnosis.md
  role: supporting
created_at: '2026-09-10T14:24:50Z'
updated_at: '2026-09-10T14:24:50Z'
platforms: []
---

# MC2 切分开销与通信等待归因

## 切分的两侧成本

MC2 增加通信与计算块数可以增加重叠机会，同时会增加通信启动、同步及流水重新填充的次数。较小 Matmul 块还可能使可调度块数不足、尾块比例上升或 Cube 利用率下降。较小通信消息可能无法维持大消息的有效带宽。

因此“通信慢就尽量多切”或“两侧时间相近则均分必然最优”都不是一般结论。应比较切分前后每侧累计工作时间、等待区间和最终关键路径；被掩盖的一侧增加成本，在仍完全被掩盖时影响较小，超过重叠余量后仍会延长整体时间。

## 隔离测量的边界

同样 shape 与数据量下，纯通信和纯计算测量可以建立瓶颈假设；按行数归一化的时间只在该测量区间近似适用，不能直接外推成固定的线性开销模型。隔离版本仍需提供合法输入并闭合同步依赖；单纯注释一个 Process 可能留下永久等待或读取未初始化数据。

当计算侧在某条流水上等待通信完成时，该流水的占用指标可能含依赖等待。较高 MTE2 占比本身不能证明 GM 搬运带宽不足。需要结合目标 profiler 指标定义、跨核等待位置及完整与隔离版本的时间线，区分有效搬运、排队和等待。固定差值阈值不是通用判定标准。

## 验证

对每个合法切分候选记录消息大小、计算块 shape、实际用核、尾块、各 rank 耗时及端到端时间。保持数据、通信拓扑、缓存策略和计时范围一致，确认瓶颈随改动变化，再解释收益。

通信时间与带宽口径参见[通信计时与带宽](communication_timing_and_bandwidth.md)。本地计算可独立时，参见[本地 Matmul 与通信依赖](../communications/local_matmul_and_communication_dependencies.md)。
