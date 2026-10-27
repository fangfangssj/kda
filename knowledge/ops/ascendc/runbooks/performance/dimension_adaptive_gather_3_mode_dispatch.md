---
type: Runbook
title: "Dimension-adaptive gather with 3-mode dispatch — detect gather dim → classify → select specialized kernel"
description: "Dimension-adaptive gather with 3-mode dispatch — detect gather dim → classify → select specialized kernel"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-FA-CV-9"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-FA-CV-9 — Dimension-adaptive gather with 3-mode dispatch — detect gather dim → classify → select specialized kernel

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；适用条件见正文（任意维度 gather/index/select 算子，按 gather 维分 3 模式派发：last_dim / permute_last_dim / scalar）。
- 验证来源：cv-agent gather_elements_v2 3-kernel 架构（last_dim / transpose / scalar + per-mode VEC 对齐策略）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`verified_on: cv-agent gather_elements_v2 3-kernel architecture (last_dim / transpose / scalar) with per-mode VEC alignment strategy`

**Pattern**: For gather/index/select ops on arbitrary dimensions, classify the gather dimension into 3 modes and dispatch to a specialized kernel per mode. Each mode has a different DataCopy strategy based on memory access pattern.

**3 modes**:
1. **last_dim** (dim == -1 or dim == ndim-1): innermost dimension → contiguous elements per row → VEC DataCopy row-by-row directly. No transpose needed.
2. **permute_last_dim** (dim != -1 but index count fits in UB): transpose input so gather dim becomes last → same VEC row-by-row kernel as mode 1. Overhead: one transpose pass.
3. **scalar** (dim != -1 AND index count too large for UB transpose, OR unaligned): element-by-element scalar gather via `DataCopy<1,1>`. Slowest but always correct.

**Mode selection logic** (host-side, in pybind11 or tiling computation):
```cpp
if (dim == ndim - 1 || dim == -1) → mode = LAST_DIM;
else if (index_count * element_size ≤ UB_SIZE / 2) → mode = PERMUTE_LAST_DIM;
else → mode = SCALAR;
```

**Why not one generic kernel**: A single gather kernel would need to handle arbitrary-dim access patterns with runtime stride computation → scalar access for ALL elements → no VEC utilization. Mode 1 gives full VEC bandwidth for the most common case (gather on last dim).

**Detection**: count per-mode kernel .cpp files. Single generic gather kernel → gap. 3 separate .cpp with mode names → pattern applied.

**Evidence**: cv-agent `gather_elements_v2/kernel/` has `gather_elements_v2_last_dim_kernel.h`, `gather_elements_v2_transpose_kernel.h`, `gather_elements_v2_scalar_kernel.h` — 3 modes with shared common kernel base class.

**Cross-ref**: CAND-FA-CV-7 (multi-strategy shape dispatch — same dispatch-by-classification pattern, different classification axis: dimension vs shape), PB-22 (DataCopy 32B alignment — scalar mode bypasses this by using DataCopy<1,1>), OL-124 (TQue<VECOUT> constraint — gather output may need TBuf not TQue for scalar mode)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-9，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
