---
type: Runbook
title: "UB-to-UB DataCopy Silent Data Corruption"
description: "UB-to-UB DataCopy Silent Data Corruption"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.data_movement, topic.memory]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-9-ub-to-ub-datacopy-silent-data-corruption.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3, "950"]
aliases: ["PB-9"]
---

# UB-to-UB DataCopy Silent Data Corruption

> 原 OKF v1 卡号：**PB-9**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - DataCopy(localDst, localSrc, count) between two LocalTensors (both in UB) silently produces garbage data. No compile error, no runtime error — just wrong values

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass ops (17_EmbeddingWithInitialLayernormBackward, 20_FusedRopeWithQkNormAndKvCacheUpdate). Do not downgrade.
- **Symptom**: `DataCopy(localDst, localSrc, count)` between two LocalTensors (both in UB) silently produces garbage data. No compile error, no runtime error — just wrong values. Discovered when LayerNorm V2 passed for norm_size ≤ 4096 (single tile) but produced ~20% mismatch with mean_abs_diff ~1.14 for norm_size > 4096 (multi-tile). Removing the UB-to-UB DataCopy and operating directly on the dequeued tensor fixed it completely.
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Never copy between LocalTensors using DataCopy. Instead:
  - Operate directly on the source tensor (e.g., run BinaryFoldReduceSum on the dequeued xd tensor)
  - Use VEC ops as a "copy": `Adds(dst, src, 0.0f, count)` if you must copy
  - Or use `Duplicate` to zero a buffer, then `Add(dst, dst, src, count)`
- **Status**: OPEN
- **Evidence**: LayerNorm V2 debugging session 2026-04-09; kernel/layernorm_kernel.h Pass 1 fix
  - op#30 NMS a3 ds kw-1 (2026-05-07): Used `Adds(dst, src, 0.0f, count)` identity copy from TQue<VECIN,1> dequeued tensors to persistent UB compute buffers, avoiding UB→UB DataCopy entirely. 31/31 bit-exact vs Python CPU reference. Confirms Adds-identity as canonical V220 workaround for VECIN→VECCALC data movement.

<!-- 迁移自 porter kb/target/ascendc/（PB-9，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
