---
type: Guide
title: Ascend C 跨核通知通道与缓冲复用
description: 将 flag 标识、未消费通知计数、生产消费流水和参与核集合分开建模，避免将计数容量误解为循环次数限制。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.pipeline
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-crosscore-sync.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/inter_core_sync/cross_core_wait_flag_isasi.md
  role: supporting
created_at: '2026-09-10T12:49:29Z'
updated_at: '2026-09-10T12:49:29Z'
platforms:
- '950'
- a2
- a3
---

# Ascend C 跨核通知通道与缓冲复用

跨核通知需要描述参与核集合、通知通道、数据区及复用期次。HardEvent 用于核内流水交接，CrossCore flag 用于受支持 Kernel 类型中的核间同步；二者的标识空间和作用域不能混用。具体 mode、pipe 与 Kernel 类型组合见 [CrossCoreWaitFlag](../../apis/simd_api/basic_api/sync_control/inter_core_sync/cross_core_wait_flag_isasi.md)。

通知生产完成和通知消费完成承担不同职责。生产者发布数据后，消费者才能使用；生产者复用缓冲前，还需知道旧数据的所有消费者已经结束访问。只有单向 ready 信号，不能自动证明可安全覆盖旧槽位。

## 标识与计数

有限 flagId 应作为显式分配的通道使用。将无界 tile 编号直接作为 flagId，可能因硬件截断与其他通道碰撞。高阶 Matmul、同步实现或嵌套组件占用哪些 ID，需结合实际实例及版本核对；单一文件搜索无命中不能证明整个调用链没有占用。

对采用计数语义的通知，容量限制针对未消费通知的积压。假设一个计数器仅允许有限积压，只要每轮发布都被对应消费，总循环次数仍可超过该容量；不能据此推导循环必须少于 16 轮等限制。反之，总 Set/Wait 次数相等也不足以证明任意时刻都不会积压溢出或互相等待。

## 流水及参与者边界

发布必须排在实际数据生产完成之后，等待必须覆盖随后首次消费数据的执行路径。指定流水上的等待不能未经验证就当作标量命令下发的门控。平台迁移时应重新确认等待的生效位置，不能仅因源平台运行通过就复用其假设。

来源对 950 无模板 Wait 与显式默认模板实参的解释存在需要底层实现进一步确认的边界；本卡不将二者写成普遍不同的 C++ 调用语义。

全体同步要求所定义的参与者都按相同协议到达；分核分支、空工作核、尾轮和提前返回都可能破坏这一条件。减少同步开销应调整合法的工作粒度或通信协议，并验证数据交接，不能直接删除必要同步。
