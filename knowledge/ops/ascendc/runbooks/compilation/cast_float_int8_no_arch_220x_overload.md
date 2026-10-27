---
type: Runbook
title: "`Cast<float, int8_t>` has no arch-220x overload — two-step int8→half→fp32 lift is exact for the whole int8 range"
description: "`Cast<float, int8_t>` has no arch-220x overload — two-step int8→half→fp32 lift is exact for the whole int8 range"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["EC-300"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-300 — EC-88: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend910_9382（arch 220x）；CANN 9.0.0；bisheng=a3 toolchain；op_class=all（任何将 int8 量化权重/激活提升到 fp32 的 kernel）。
- 验证来源：2_FFN 伪量化路径（2026-07-25，worker iter 1 编译期捕获）。

**EC-300** — `Cast<float, int8_t>` has no arch-220x overload: compile-LOUD twin of PB-37's silent uint8 garbage (candidate table: int8→half/int16 only). Fix = two-step int8→half→fp32 (exact for whole int8 range). Don't shop RoundModes — the pair is absent. (2_FFN pquant, 2026-07-25)

- Former A5Ops ID: `EC-88`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend910_9382 (arch 220x); cann=9.0.0; bisheng=a3 toolchain; op_class=all (any kernel lifting int8 quant weights/activations to fp32)`
`verified_on: 2_FFN pseudo-quant path (2026-07-25, compile-caught at worker iter 1)`

- **Symptom**: `Cast(fp32_dst, int8_src, ...)` fails at COMPILE time — no `CastIntrinsicsImpl` overload; the compiler's own candidate table lists only `int8 → half` / `int8 → int16` from an int8 source. This is the loud twin of PB-37 (`Cast<float, uint8_t>` on 351x silently produces garbage) — same missing-pair class, opposite failure mode (compile error vs silent miscompile), so on 220x you get stopped at build instead of poisoned at runtime.
- **Fix — canonical two-step lift through half**:
  ```cpp
  Cast(tmp_half, src_i8,  RoundMode::CAST_NONE, count);  // int8 → half  (exact: −128..127 representable)
  PipeBarrier<PIPE_V>();
  Cast(dst_f32,  tmp_half, RoundMode::CAST_NONE, count); // half → fp32  (exact)
  PipeBarrier<PIPE_V>();
  ```
  Both legs are in the supported pair table; the chain is exact for the whole int8 range.
- **Detection**: any `Cast<…float…, …int8_t…>` in a 220x kernel TU → expect the no-overload error. Do NOT shop RoundModes — the (src,dst) pair itself is absent from the table.
- **Cross-ref**: PB-37 (the uint8 silent-garbage sibling on 351x + the same two-step workaround shape), P-P46 (the requant cast chain this feeds into for A8W8-style quant pipelines). backend=ascendc.
