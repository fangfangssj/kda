---
type: Guide
title: 多阶段 Cube–Vector 流水的任务与槽位生命周期
description: 说明预加载、消费和回收游标，AIC/AIV 分工，以及 Flash Attention 多阶段流水的启动与排空条件。
tags:
- task.design
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P14_cv_pipeline_preload.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P17_cv_parallel_block_sizing.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P19_custom_pingpong_double_buffer.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P20_triple_buffer_rotation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P21_matrix2x2_buffer_policy.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P40_workspace_double_buffer.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P51_dynamic_aic_aiv_ratio.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P73_cv_parallel_aic_aiv_ratio.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P75_dual_aiv_workload_split.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P80_sink_task_pipeline_injection.md
  role: supporting
created_at: '2026-09-10T13:32:30Z'
updated_at: '2026-09-10T13:32:30Z'
platforms: []
---

# 多阶段 Cube–Vector 流水的任务与槽位生命周期

Cube–Vector 流水把不同任务的矩阵计算、向量处理及数据搬运交错执行。每个任务仍需按自身的数据依赖完成各阶段；环形槽位只是保存不同任务的物理空间，不是完成信号。

对每个槽位，应能区分“可生产”“数据已发布”“消费者正在使用”“所有消费者已完成”四种状态。生产、消费与回收游标可以不同，但再次覆盖一个槽位前必须等到上轮使用结束。三槽、四槽、二乘二索引或简单 `loop % depth` 本身都不证明这一点。事件方向应由真实生产流水与消费流水确定，不能仅按 Buffer 类名或示例注释套用。

## 启动、稳态与排空

流水启动时，后续阶段可能还没有任务；排空时，不能继续等待已经没有生产者的新任务。`isValid` 应与任务身份和发布/消费事件一致，不能通过标记无效绕过仍有参与者等待的同步。验证零任务、单任务、少于槽位深度、多次轮转以及最后一个尾任务。

Flash Attention 的 S2 分块流水除矩阵中间值外，还包含 online softmax 的跨块状态。max/sum 与输出缩放因子应属于同一 query 范围和同一次更新。新增 sink 或 sparse 任务时，需保持有效 token 集合、mask 和状态更新顺序；不能把重复出现的 token 计算两次，或在跳块时跳过必需的事件消费。

GM workspace 要为每核、每槽、每阶段明确定义字节范围。增加槽位可缓解等待，也会增加空间和启动成本，不能由两份改成四份就断言完全并行。更多缓冲也不能消除真正的同任务依赖。

## 分工与粒度

增大 query 块或 KV 块可能提高矩阵效率，但同时增加中间值、L1/UB 和向量阶段的工作量。应结合各阶段的实际耗时与资源峰值调整粒度，不能只比较 Cube 的利用率。

改变 AIC:AIV 模式后，编译实例、Host 启动、核编号、工作区划分及同步参与者必须一致。将 M/S1 的不同行分给多个 AIV，需要无遗漏、无重叠地处理尾行，未分到工作但参与同步的核也要遵守协议。来源中的 K=2048、16 行边界、固定核数和吞吐翻倍都不是通用模式选择规则。

验证以完整时延和逐阶段流水为依据，并覆盖不同核配比、尾块和多轮槽位复用。基础约束见[执行模式与 Host 合同](execution_mode_and_host_contracts.md)、[跨核通道复用](../synchronization/cross_core_channels_and_reuse.md)和[缓冲所有权](../memory/buffer_ownership_and_slot_layout.md)。
