---
type: Optimization
title: "SIMD vs SIMT selection"
description: "Core rule: Prefer SIMD unless scatter-write (atomicAdd) is required Scenario Choice Reason ------------------ scatter-write (atomicAdd to random addresses) SIMT SIMD's SetAtomicAdd requires alignment"
tags: [task.optimization, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.programming_model.execution_model.simt, topic.implementation.atomic]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p9-simd-vs-simt-selection.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

# SIMD vs SIMT selection

> 原 OKF v1 卡号：**P-P9**。

## 优化点

**Core rule**: **Prefer SIMD unless scatter-write (atomicAdd) is required**

| Scenario | Choice | Reason |
|------|------|------|
| scatter-write (atomicAdd to random addresses) | **SIMT** | SIMD's SetAtomicAdd requires alignment and is functionally limited |
| indirect-read + weighted sum (e.g., SG Forward) | **SIMD** | DataCopy block transfer + 4-pipeline parallelism |
| Contiguous aligned read/write | **SIMD** | Natural pipeline-overlap advantage |
| scatter-read + scatter-write mixed | **SIMT** | SIMD cannot orchestrate when both ends are irregular |

**~~Old rule~~ (deprecated)**: ~~"indirect indexing / random access -> SIMT"~~

**Reason for deprecation (msprof evidence)**: SG Forward has indirect addressing (each token reads a different expert), but the expert rows themselves are contiguous in memory. SIMD DataCopy batch-transports expert rows to UB (via MTE2), which is 2-7x faster than SIMT thread-scalar scattered reads. SIMT's VEC pipe gets stalled by GM read latency (vec=0.95+, mte2=0.000); SIMD runs 4 pipes simultaneously (vec+scl+mte2+mte3 each at 30-90%).

**Key insight**: "Indirect addressing" must distinguish the addressing layer from the data layer. SG Forward's addressing layer is indirect (which expert), but the data layer is contiguous (expert rows are contiguous memory). SIMD's DataCopy handles the data layer while scalar computes the addressing. indirect-read != must-be-SIMT.

**SIMT architecture limitation**: In SIMT mode, GM access only uses the VEC pipe's load/store units; the MTE2/MTE3 DMA engines do not participate. 4-pipeline parallelism is unachievable — this is a hardware limitation, not a code problem.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/kernel_launch.md（P-P9，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `platform_compat`；trigger: algorithm classification；优先级: HIGH。

> 索引行原文：`| P-P9 | **SIMT vs SIMD decision framework** | platform_compat | algorithm classification | **HIGH** |`

## 更新 (2026-09-19)

私有沉淀仓（不公开，pinned commit cb9c78c85ead3088398f3455051355349298b3cb）SIMT_VS_SIMD_DECISION.md#pattern-p-p9-summary 节选卡同 lesson 复核，补录其相对 v1 卡的两处增量：（a）默认值补全——group-local 依赖且 `group < 256` → **SIMT**（v1 卡表格未含此条；同内容见 `../concepts/simt_vs_simd_decision.md` Decision Tree Step 3a/3b）；（b）强制验证纪律——选型只是假设：必须 A/B benchmark 确认且两次测量在同一 NPU 同一 session（OL-27）；SIMD 不得降低精度（OL-30），faster-but-wrong 的 SIMD 路径直接出局；若 SIMD 比 SIMT 慢就直接用 SIMT，不再继续投入优化 SIMD 变体。
