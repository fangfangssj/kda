---
type: Guide
title: SIMT 共享存储与同步作用域
description: 区分块内 UB 共享、跨核 GM 交换和函数串行，按真实存活资源与生产消费关系判断可见性及复用。
tags:
- task.design
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-simt-best-practices/references/intra-core-shared.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-simt-best-practices/references/data-transfer.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-simt-best-practices/references/multi-core-sync.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-simt-tiling-design/references/guide.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/programming_guide/advanced_programming/advanced_ai_core_programming_model/simd_simt_hybrid_programming/memory_hierarchy.md
  role: constraint
created_at: '2026-09-10T13:06:26Z'
updated_at: '2026-09-10T13:06:26Z'
platforms:
- '950'
---

# SIMT 共享存储与同步作用域

SIMT 可以直接访问 GM，也可以使用块内共享的 UB。是否暂存数据取决于重用次数、访问模式、同步开销和容量，不能把“不使用 DataCopy 的逐元素示例”推广为 SIMT 永远不涉及 UB。

共享地址不自动提供访问顺序。多个线程读取其他线程写入的 UB 前，需要满足目标接口的块内同步和可见性约束；参与屏障的线程必须一致到达相应阶段。多个线程写同一地址时，应先明确算法需要唯一写者、归约还是原子更新。原子操作只在其语义与算法一致时成立，不能用于掩盖错误的输出映射。

## 三类不同的依赖

| 数据交接 | 要证明的条件 |
|---|---|
| 同一块内线程经 UB 交换 | 地址有效、生产完成、参与线程及屏障范围匹配 |
| 同一核块连续执行 VF | 调用顺序与相关流水完成语义匹配，后续不会提前复用缓冲 |
| 不同核块经 GM workspace 交换 | 发布/等待协议覆盖所有实际参与核，数据可见，下一期复用不会覆盖未读数据 |

源码中两个 VF 调用串行，不表示其他核块也已到达同一阶段。跨核同步还涉及调度模式、工作核数量和目标产品支持；不能把任意位置添加 `SyncAll` 或单一 SetScheduleMode 示例作为通用修复。条件分支或空工作核导致参与者缺席时，等待可能无法完成。

UB 预算应包括同时存活的 SIMD 缓冲、SIMT 静态/动态共享区、运行保留区及所选缓存配置。[混合编程存储层级](../programming_guide/advanced_programming/advanced_ai_core_programming_model/simd_simt_hybrid_programming/memory_hierarchy.md)描述了这些区域共享容量的关系；其中容量数值具有产品和配置范围，不能固定把 216 KiB 当作所有执行模式的可用 UB。

采用动态数组或 LocalMemAllocator 时，需遵守各自的分配协议，不能把同一动态区基地址交给两个互不知情的分配器。申请范围不足可能影响其他 UB 区域，而非只在本数组中产生错误。

验证应覆盖单块与多块、完整和尾部数据、多个连续调用及不同缓存/共享内存配置。增加屏障后偶然通过只能说明时序变化，需要进一步证明同步范围与缓冲生命周期正确。
