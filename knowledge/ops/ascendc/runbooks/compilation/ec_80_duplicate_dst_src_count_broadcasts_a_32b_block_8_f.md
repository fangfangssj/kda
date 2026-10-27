---
type: Runbook
title: "`Duplicate(dst, src, count)` broadcasts a 32B BLOCK (8 fp32 / 16 fp16), NOT a single scalar — compiles clean but produces garbage output"
description: "<!-- applies_to_backend: ascendc -->"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.vector, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-80-duplicate-dst-src-count-broadcasts-a-32b-block-8-f.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: ["950"]
aliases: ["EC-80"]
---

# `Duplicate(dst, src, count)` broadcasts a 32B BLOCK (8 fp32 / 16 fp16), NOT a single scalar — compiles clean but produces garbage output

> 原 OKF v1 卡号：**EC-80**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - kernel compiles and launches clean. At runtime, output values are completely wrong — e.g., a GroupNorm+SiLU kernel where all D>1 output slices are identical (ma

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=elementwise,normalization`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A2/A3 — Duplicate semantics are general AscendC API, not arch-specific)`

- **Error pattern**: kernel compiles and launches clean. At runtime, output values are completely wrong — e.g., a GroupNorm+SiLU kernel where all D>1 output slices are identical (matching only the last d slice's reference), or scalar-broadcast produces garbage ~1e5× the expected magnitude.
- **Root cause**: `Duplicate(dst, src, count)` replicates an entire **32-byte BLOCK** (8 fp32 elements, 16 fp16/bf16 elements) at a time, NOT a single scalar. For fp32, `Duplicate(dst, scalar_src, 1)` copies 8 consecutive fp32 values starting from the `scalar_src` UB address — the intended scalar plus 7 garbage neighbors. When used in a per-d loop to broadcast per-channel scale/bias, every output slice gets the same block of garbage data.
- **Fix** (scalar broadcast alternatives):
  ```cpp
  // BEFORE (WRONG — Duplicate broadcasts 8 fp32 values, not 1):
  Duplicate(scaleLocal, gammaLocal[d], 1);   // copies gamma[d] + 7 garbage neighbors

  // AFTER (correct — SIMD Brcb scalar broadcast):
  // Option A: Brcb (block-broadcast scalar to all vector lanes)
  Brcb(scaleLocal, gammaLocal[d], hwNumAligned_);

  // Option B: Inline scalar × vector (preferred when you already have a Mul/Add op):
  float scale = gammaFp32.GetValue(c);
  Muls(tmpFp32[ubOff], xFp32[ubOff], scale, hwNum_);  // scalar × vector in one op
  // No Duplicate needed — AscendC SIMD Mul/Add accept scalar right-hand-side directly.
  ```
- **Key insight**: AscendC SIMD VEC ops (`Muls`, `Adds`, `Divs`, `Subs`, `Mins`, `Maxs`) accept a **scalar** second operand directly — no need to broadcast the scalar into a tensor first. The `Duplicate` API exists for replicating multi-element BLOCKS (e.g., duplicating a row of weights across a batch), not for scalar→vector conversion.
- **Evidence**: group_norm_silu precision fix (2026-06-26, A5/Ascend950PR, CANN 9.0.0): D>1 mode used `Duplicate` to broadcast per-channel gamma/beta scalars into UB tensors. All D output slices were identical to the last d's reference because Duplicate copied 8-element blocks containing garbage. Fixed by removing Duplicate entirely and using inline `Muls`/`Adds` with float scalars directly.
- **Other instances (predicted)**: any kernel that uses `Duplicate` intending to broadcast a single scalar — per-channel affine transforms, bias-add loops, per-head attention scaling, per-expert MoE gating. The fix is always: either use SIMD VEC ops with scalar RHS (no broadcast needed), or if a full tensor IS needed, use `Brcb` for proper scalar-to-vector broadcast.
- **Related**: OL-260 (member shadowing in Init() — the SAME group_norm_silu session had BOTH bugs; Duplicate was a red herring once the shadowing fix was in), P-P4 (Dynamic block size), `ASCENDC_API_CATALOG.md` (Duplicate API signature).

<!-- 迁移自 porter kb/target/ascendc/（EC-80，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
