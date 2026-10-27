---
type: Guide
title: DMA 原子写回的状态范围与输出初始化
description: 说明随路原子累加影响后续 GM 搬运、需要正确初始化目标与关闭状态，并区分 DMA 原子和跨卡通信原子。
tags:
- task.implementation
- topic.implementation.atomic
- topic.data_movement
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-atomic.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/atomic_operations/set_atomic_add.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/atomic_operations/disable_dma_atomic.md
  role: constraint
created_at: '2026-09-10T12:27:52Z'
updated_at: '2026-09-10T12:27:52Z'
platforms:
- '950'
- a2
- a3
---

# DMA 原子写回的状态范围与输出初始化

`SetAtomicAdd<T>` 配置后续目的地址为 GM 的受支持搬运执行旧值加新值；它不只改变紧随其后的一次函数调用。完成需要累加的搬运阶段后应配对 `DisableDmaAtomic`，使后续输出恢复普通写入。

原子加需要一个已定义的基值。目标 GM 不会因开启原子加自动清零；若采用清零后累加或某个写者先覆盖再累加，初始化必须在所有累加写者开始前完成。仅让某个核先执行一条普通写指令，并不构成跨核的先后保证。

多写者冲突并不总是加法归约。应先确认业务语义需要加、max/min，还是本应由一个写者独占的覆盖操作，再选择目标平台和 dtype 支持的机制。T 必须与真实输出数据格式匹配。

浮点原子累加的结果可能因写入顺序而变化。要求确定顺序时，需要明确规定并实现全局合并顺序；单核的流水屏障不能独立保证其他核或 rank 的累加次序。

DMA 随路原子、L0C 内部累加与 Hcomm 通信原子位于不同层次，不能互相替代。某种 DMA 原子支持，也不能证明任意跨卡路径都继承该行为。

验证应覆盖非零初始输出、重复调用、后续普通写入和多写者并发，检查数值误差与状态是否泄漏。具体搬运方向、dtype 与版本范围以 [SetAtomicAdd](../../apis/simd_api/basic_api/atomic_operations/set_atomic_add.md)和 [DisableDmaAtomic](../../apis/simd_api/basic_api/atomic_operations/disable_dma_atomic.md)为准。
