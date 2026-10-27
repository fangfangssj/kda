---
type: Guide
title: 按实际代价划分不均匀工作量
description: 用可获得的稀疏度、有效长度和对齐开销估算工作量，区分行数均分、代价均分与实际核间耗时。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P16_cost_driven_core_partition.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P4_load_balance.md
  role: supporting
created_at: '2026-09-10T13:19:40Z'
updated_at: '2026-09-10T13:19:40Z'
platforms: []
---

# 按实际代价划分不均匀工作量

行数或 tile 数均分只有在每项代价接近时才容易获得负载均衡。稀疏行的 nnz、变长序列、mask 跳过的块，以及尾部的实际计算/搬运量，都可能使相同数量的工作项具有不同耗时。已有 [按行均分的 nnz 倾斜案例](../../runbooks/performance/tiling_nnz.md)提供了逐核观测的具体例子。

可对工作项建立非负代价并按累积代价划分连续区间，以保留一定的地址局部性。模型应包含实际有效工作和不能被尾块消除的固定开销；数据逻辑上更短，不一定使硬件指令数量同比下降。参考中的对齐常数、轴权重 6/10 或差异小于 20% 都属于特定启发式，不能作为硬件定律或通用验收门槛。

代价信息需要在执行该分配的阶段可获得。Host 只有 shape 而没有 Device 上的稀疏索引时，不能直接假设 tiling 已掌握每行 nnz；读取或预处理这些元数据的开销，也应计入设计。

## 分配需要保持的语义

每个工作项应被完整分配，区间不能重叠或遗漏。总预测代价为零、单项代价超过平均份额、尾部取整和整数乘法溢出都需要明确行为。单个巨大工作项可能成为无法通过换核号消除的长尾；进一步拆分它会引入部分结果及合并责任。

分核仍需满足输出所有权与同步参与范围。按代价搬动任务、按稀疏 mask 跳块或改变 AIC/AIV 分工后，应重新验证生产消费关系，不能只调整 `blockIdx` 公式。二维网格的除法/取模编号本身也不构成对角线调度或热点消除。

校准时同时比较预测代价、逐核实际耗时和整体时延。若预测均衡而耗时仍倾斜，应继续检查访存冲突、缓存行为、同步或计算阶段差异。负载更均匀也可能因额外调度成本而变慢，最终收益以同输入、同语义的完整执行为准。
