---
type: Runbook
title: "`Cast<T, T>` same-dtype is a no-op — destination stays uninitialised (silent data corruption)"
description: "`Cast<T, T>` same-dtype is a no-op — destination stays uninitialised (silent data corruption)"
tags: [task.debugging, topic.data_movement, topic.memory, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-36-cast-t-t-same-dtype-is-a-no-op-destination-stays-u.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-36"]
---

# `Cast<T, T>` same-dtype is a no-op — destination stays uninitialised (silent data corruption)

> 原 OKF v1 卡号：**EC-36**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - No compile error. At runtime, Cast(dstLocal, srcLocal, RoundMode::CAST_NONE, count) where dstLocal and srcLocal have the SAME dtype T produces no hardware instr

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: No compile error. At runtime, `Cast(dstLocal, srcLocal, RoundMode::CAST_NONE, count)` where `dstLocal` and `srcLocal` have the SAME dtype T produces no hardware instruction. `dstLocal` retains whatever was in that UB region from prior kernel state (garbage / zeros / stale prior-tile data). Downstream ops using `dstLocal` silently use uninitialised values.
- **Root cause**: Cast's codegen optimises same-dtype casts to nothing (compiler assumes no conversion needed). Valid for in-place casts; catastrophic when the intent was to copy from src to dst.
- **Fix (two options)**:
  1. **Per-dtype Compute specialisation**: template-specialise so the fp32 path skips the Cast entirely and operates directly on the dequeued input tensor.
  2. **Explicit copy via arithmetic no-op**: `Adds(dst, src, 0.0f, count);` forces a real data movement. Or `Muls(dst, src, 1.0f, count);`.
- **Detection signal**: kernel produces output that looks like "some operands missing" — e.g., `out ≈ a + 0 · b` or `out ≈ b` when formula should be `out = c*a - d*b`. Hint: if tail of your compute chain uses a buffer that was "copied via Cast<T,T>", suspect this.
- **Prevention (Phase B checklist)**: any `Cast<T1, T2>` where T1 == T2 — audit. If intent is "copy from input to scratch", use `Adds(..., 0.0f, ...)` or avoid the scratch entirely.
- **Evidence**: op#16 Batched2DRopePositionEncodingBackward Phase D iter 2 (2026-04-22). fp32 path `Cast(gcF, gc, CAST_NONE, count)` where both are fp32 left `gcF` uninitialised → downstream `Mul(prodA, gcF, sinF)` produced `prodA ≈ 0` → `out = prodB - prodA = gs*cos(θ)` consistently (missing the `-gc*sin` term). Fix: switched fp32 path to skip Cast, operate on dequeued input directly. 50/50 PASS after.
- **Related**: P-P52 fp32 promotion — when promoting bf16/fp16 → fp32, the Cast IS needed (different dtype). Trap is only when dtypes happen to match.

<!-- 迁移自 porter kb/target/ascendc/（EC-36，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
