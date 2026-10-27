---
type: Guide
title: 分块流水的启动、事件复用与排空
description: 补齐 CopyIn/Compute/CopyOut 流水中的零任务、槽位归还、跨 Scalar 消费和事件资源生命周期。
tags:
- task.design
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P1_double_buffer.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P5_pipeline_sync.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P1_double_buffering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P5_pipeline_sync.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P1.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P5.yaml
  role: supporting
created_at: '2026-09-11T01:38:31Z'
updated_at: '2026-09-11T01:38:31Z'
platforms: []
---

# 分块流水的启动、事件复用与排空

对 L 个有效 tile 的二槽流水，可以先发起 tile0，再在处理 tile i 的同时为 i+1 准备空闲槽，最后消费尾块。L=0 不发起 CopyIn(0)；L=1 只执行一次完整任务；L>1 才有相邻 tile 的重叠机会。嵌套循环可以逐内循环排空，也可以在证明任务地址/状态正确后跨外层预取，不能只用“必须只改内循环”的流程约定限制知识。

每个输入槽从空闲→搬入中→就绪→计算消费→空闲，输出槽从空闲→计算写→可搬出→搬出中→空闲。输入不能在计算完成前重写，输出不能在 MTE3 读取完成前重写。Alloc/EnQue/DeQue/Free 的队列协议与手工事件方案应分别核对；源码出现 CopyIn(i+1) 或 BUFFER_NUM=2 不等于已经实现重叠。

TQue depth 与 InitBuffer 的物理份数不同；双缓冲不要求所有队列包括 L0C 全部改成2，也不能反向推出 L0C 永远只能1。链式 Mmad 对同一个 partial 的累计需保持状态，独立 tile 的 L0C 轮转另需资源和消费协议。tileSize 减半只在相关按 tile 线性增长的区域同时翻倍时近似维持容量，固定scratch与不同队列份数仍需逐项计算。

典型交接包括 MTE2→Vector、Vector→MTE3、Vector结果→Scalar读取，以及Scalar写入→Vector消费；事件类型由实际操作与存储依赖决定。Cast 本身不是MTE2，Matmul完成也不能用PIPE_V独立保证。同一流水内部是否需屏障还取决于接口和编译器保序条件，见[精度链同步](precision_chain_dependency_scope.md)。

事件需要分配、初态、发布、等待消费、复用和最终释放。可在上一期配对已经完成后复用同一逻辑资源；不可无条件重复Fetch却不考虑分配器生命周期，或覆盖仍活跃的事件。不同HardEvent是否共享编号空间、可用数和保留槽由SDK及平台决定，不能将来源“全局8个”或“类型不同绝不冲突”作为统一规则。

Wait应位于对应生产完成标记之后并在消费之前；把Wait更早下发通常影响重叠，不能仅凭源码距离推断读到未完成数据。Set/Wait静态次数相等也不证明条件分支、首轮和尾轮中的实际配对。移除跨核SyncAll前，必须证明其全部参与者和跨核依赖被另一合法协议覆盖。

验证0/1/2/多tile、每个槽、内外循环边界、尾块、标量归约读回、重复执行和最后一次搬出。吞吐由各阶段时间、共享带宽和资源竞争决定；一阶段较长不代表任何重叠都毫无收益，阈值和速度必须来自配对测量。
