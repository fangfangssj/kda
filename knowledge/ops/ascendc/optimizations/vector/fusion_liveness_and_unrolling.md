---
type: Optimization
title: VF 融合、存活寄存器与循环展开的取舍
description: 从实际中间值写读、活跃状态和指令依赖评估融合及展开，保留尾部与数值等价条件。
tags:
- task.optimization
- topic.memory
- topic.parallelism
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/R1_vf_fusion.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/R2_register_reuse.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/R3_dual_issue.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/R7_simd_simt_hybrid.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/performance-issues-aiv.md
  role: supporting
created_at: '2026-09-11T01:27:48Z'
updated_at: '2026-09-14T04:19:37Z'
platforms: []
---

# VF 融合、存活寄存器与循环展开的取舍

若前一个 VF 把中间值写入 UB，后一个 VF 立即读回，且两阶段可在同一调用、同一有效域内执行，融合可让中间值留在寄存器。例如 `(a+b)*c` 可在一次加载后的计算链中完成，消除中间量的一次 UB Store 和一次 Load。这是该中间值的访问减少，不是整个算子必然获得 2–5 倍收益。

融合会改变同时存活的数据集合。源例从分别处理 Add/Mul 的逻辑变量，变为同时保留 a/b/c、临时和输出；编译器仍负责物理寄存器分配，不能用源码变量个数直接断言实际寄存器数或 spill。原地复用 tmp 可缩短一些值的存活区间，但会延长依赖链；只有旧值已不再使用、重叠被接口允许时才可复用。

两次循环展开可显式暴露两个独立 tile 的加载、计算和写回，给编译器更多调度机会。需分别推进 offset 和 mask，处理 `repeats/2` 个成对块及 `repeats%2` 尾块；如果 UpdateMask 修改 remaining，两个 mask 必须对应连续但不同的有效区段。相邻的两条 Load/Compute 不因此必然双发射，目标流水能力、发射限制和寄存器压力要看实际产物与量测。

SIMD 规则向量链与 SIMT 不规则 Gather 可作为不同阶段候选：先按索引收集，再规则计算，需要计算额外暂存、调用切换、索引读取和完成交接的成本。SIMT 并不自动使随机访问变成连续访问。是否跨 VF 或混用模型，应按目标 SDK 的启动与内存合同核对，参见[SIMT 共享存储与同步](../../guides/simt/shared_memory_and_sync_scope.md)。

保持 dtype、舍入、mask、广播坐标、状态生命期和可观察中间结果的语义；跨调用状态不能只留在临时寄存器中。布尔代数化简适用于其相应的逻辑语义，不能据此任意重排浮点加乘。测量融合前后的 UB 访问、spill/fill、代码量和整体时间，分别验证完整块及尾块。

来源把相关策略标为 retired，并称 RegTensor、VF 等 API 不存在；该元数据不能推翻目标 SDK 已有的 RegBase/SIMT 能力。本文只保留机制，具体声明、调用和重载由[RegBase 数据流](../../guides/regbase/dataflow_layers.md)及实际接口确定，不照搬跨版本骨架、统一 32 寄存器限制或 1.3–1.8 倍双发射承诺。

## 将行循环移入 VF 的完整工作域

减少调用次数还可以把外部逐行循环放进一次 VF，由 VF 内部处理本核当前 UB tile 中的多行。此时同时改变 blockDim、每核行数和每次调用行数，可能减少参与核而增加每核工作；需要一起评估总时延，不能只最小化 vf_total_executions。一次 VF 也不能访问尚未搬入 UB 的其他 GM tile。

每行重新初始化剩余元素数，按实际源、目的行距推进指针，保证前一行耗尽的 UpdateMask 不会让后一行全部失效。最末一行及列尾部独立约束；将行数窄化到 uint16_t 之前必须证明其值域。对形如 a[r]*B[r,c] 的广播，a[r] 可以在该行的列循环中保留在寄存器并复用，不应从“Load 广播指令”推断每一段列都必需重读 a[r]。向量宽度随 dtype 和接口变化，不能固定为 64 个逻辑元素。

来源给出的调用次数阈值与合并骨架只用于定位候选。保留列循环后的跨行状态边界，并观察融合后的寄存器存活、代码量、spill 和核间分布；一次调用更长可能掩盖并行度下降。
