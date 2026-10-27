---
type: Runbook
title: "SIMD binary-scalar VEC ops (Divs/Muls/Adds/Subs/Sub/Add/Mul/Div) reject int32, int16, int8, bf16 — supported dtype list verified"
description: "SIMD binary-scalar VEC ops (Divs/Muls/Adds/Subs/Sub/Add/Mul/Div) reject int32, int16, int8, bf16 — supported dtype list verified"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.vector, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-23-simd-binary-scalar-vec-ops-divs-muls-adds-subs-sub.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3, "950"]
aliases: ["PB-23"]
---

# SIMD binary-scalar VEC ops (Divs/Muls/Adds/Subs/Sub/Add/Mul/Div) reject int32, int16, int8, bf16 — supported dtype list verified

> 原 OKF v1 卡号：**PB-23**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Building a kernel that uses AscendC::Divs<int32_t>(dst, src, scalar, count) (or any of Muls / Adds / Subs / Sub / Add / Mul / Div with int32) fails at compile w

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Severity**: HIGH (silent compile-time rejection — bisheng `static_assert` halts the build with no fallback)
- **Status**: CONFIRMED 2026-04-30 op#22 Nonzero V4 kw-5 build iter 1
- **Affected**: Ascend950PR / CANN 9.0.0 b103 / bisheng 2026-03-21. Likely V220 too (not yet checked).
- **Symptom**: Building a kernel that uses `AscendC::Divs<int32_t>(dst, src, scalar, count)` (or any of `Muls / Adds / Subs / Sub / Add / Mul / Div` with int32) fails at compile with the bisheng error: `static_assert(SupportType<T, half, float, int64_t, uint64_t, complex32, complex64>())`. The supported dtype set for these binary-scalar VEC ops is exactly `{half, float, int64_t, uint64_t, complex32, complex64}` — int32 is NOT in the list, nor are int16, int8, or bf16 (bf16 must Cast→fp32 first per PB-4).
- **Workaround (canonical)**: for index arithmetic, use `int64_t` for the SIMD vector buffer dtype. The supported-list includes `int64_t` and is verified working on a5 for Adds/Muls/Sub. If a 32-bit integer path is genuinely needed (e.g. memory-pressure on tile sizes), the upstream operation must use a Cast to fp32 first, then back to int via floor/round Cast — but be aware of fp32's 24-bit mantissa precision limit (indices > 2^24 = 16M lose accuracy).
- **Workaround for index ops specifically**: in V4 GatherMask + N-D-decode flows (P-P80), promote the per-element packed positions from int32 (output of GatherMask) to int64 via Cast<int64,int32> immediately, then do all `Divs<int64>`, `Muls<int64>`, `Sub<int64>` on the int64 buffer. This is the verified pattern from op#22 V4 kw-5.
- **Why this matters for skill briefs**: when designing SIMD vector decode for nonzero / scatter / index ops, do NOT assume int32 is supported. The static_assert is silent until you instantiate the template with int32, then halts the build. **Workers grepping ASCENDC_API_CATALOG.md for "Divs" find no dtype list and assume general support — file the dtype list in the catalog explicitly.**
- **Evidence**: op#22 22_Nonzero V4 kw-5 (2026-04-30) — initial V4 design used int32 buffers for `pos_local` (GatherMask output) and chained `Divs<int32_t>` / `Muls<int32_t>` / `Sub<int32_t>` for N-D index decode. Build failed with the static_assert. Single-fix iter switched all 5 SIMD-decode buffers to int64 → build PASS first try, V4 50/50 PASS + det 50/50.

<!-- 迁移自 porter kb/target/ascendc/（PB-23，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
