---
type: Optimization
title: "Per-uid 8-aligned GM scalar slot (multi-core scalar write)"
description: "Trigger: SIMD kernel where each AIV core needs to write a single scalar value (per-block mean / rstd / sum / count) to a shared GM array indexed by core uid. Problem: A5 DataCopy(GM, UB, count) has a"
tags: [task.optimization, topic.implementation.atomic, topic.data_movement.data_copy, topic.memory]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p49-per-uid-8-aligned-gm-scalar-slot-multi-core-scal.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Per-uid 8-aligned GM scalar slot (multi-core scalar write)

> 原 OKF v1 卡号：**P-P49**。

## 优化点

**Trigger**: SIMD kernel where each AIV core needs to write **a single scalar value**
(per-block mean / rstd / sum / count) to a shared GM array indexed by core uid.

**Problem**: A5 `DataCopy(GM, UB, count)` has a 32B minimum granularity:
- fp32: writes a block of 8 elements
- fp16/bf16: writes a block of 16 elements

If multiple cores write to adjacent indices in a GM array (e.g., `out[uid] = value`),
each core's DataCopy writes its **entire 8/16 element block**, which **overwrites
neighboring cores' data**. Result: non-deterministic / partially zero output.

**Solution**: Allocate per-uid **isolated 8-aligned slots** in GM:
```cpp
// Host: allocate (num_uids * 8) elements instead of num_uids
auto mean_buf = torch::zeros({num_uids * 8}, ...);

// Kernel: each uid writes to its own dedicated slot
DataCopy(meanGm[uid * 8], localMean, 8);  // 8 = full alignment block

// Host extracts every 8th element via stride select:
auto mean = mean_buf.view({num_uids, 8}).select(1, 0).contiguous();
```

**Why it works**: Each core's write block (8 elements) lands in a non-overlapping
region. The first element in each block is the actual scalar; the other 7 are zero
padding. Pybind layer extracts the first element via `.select(1, 0)`.

**Cost analysis**: 8x GM bandwidth for the per-uid output (e.g., for 64 cores writing
mean+rstd, this is 64 * 8 * 4B * 2 = 4KB instead of 512B). Negligible vs the
input/output tensor sizes.

**When to use**:
- ✅ Per-block scalar reduction outputs (mean, max, count)
- ✅ Norm-style kernels with per-group statistics
- ✅ Any pattern where each core produces a single scalar destination

**When NOT to use**:
- ❌ When per-core output is already ≥ 8 elements (no race)
- ❌ When using WorkspaceMerge / accumulate-into-shared (different pattern)
- ❌ When atomicAdd is acceptable (use SetAtomicAdd directly)

**Related**: OL-70 (root cause), PB-9 (related UB→UB DataCopy pitfall), P-P25 (atomicAdd-based scatter)

**Evidence**: 2_GroupNormSwish (Level 2, 2026-04-15): mean/rstd outputs are per (N, group) scalars.
Initial direct write to `meanGm[uid]` produced sporadic zero outputs across cases. After per-uid
8-aligned slot conversion: 50/50 PASS, 2.05x speedup. Validated on fp16/fp32/bf16, 2D-6D shapes.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P49，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `memory_access`；trigger: SIMD multi-core, each core writes 1 scalar to shared GM array (mean/rstd/sum, etc.)；优先级: HIGH。

> 索引行原文：`| P-P49 | Per-uid 8-aligned GM scalar slot | memory_access | SIMD multi-core, each core writes 1 scalar to shared GM array (mean/rstd/sum, etc.) | **HIGH** |`
