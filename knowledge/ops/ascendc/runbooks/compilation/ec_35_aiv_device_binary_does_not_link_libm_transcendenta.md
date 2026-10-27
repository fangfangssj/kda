---
type: Runbook
title: "AIV device binary does not link libm transcendentals — split SIMT + SIMD for per-element trig/exp/log"
description: "AIV device binary does not link libm transcendentals — split SIMT + SIMD for per-element trig/exp/log"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.programming_model.execution_model.simt, topic.data_movement.data_copy]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-35-aiv-device-binary-does-not-link-libm-transcendenta.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-35"]
---

# AIV device binary does not link libm transcendentals — split SIMT + SIMD for per-element trig/exp/log

> 原 OKF v1 卡号：**EC-35**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  ld.lld: error: undefined symbol: cosf
  >>> referenced by <kernel>.cpp:<line>
  >>>    .../device_aiv.o:(...<kernel>_vf..._simt_entry)
  ld.lld: error: undefined symbol: sinf
  ...
  ```
  Appears at the AIV device-object link step (`ld.lld -m aicorelinux`), NOT at compile. Applies identically to `std::cos` / `std::sin` / `__builtin_cosf` / `__builtin_sinf` / `expf` / `logf` / `sqrtf` / `tanf` / `powf` — all resolve to libm names at link.
- **Root cause**: AIV-only device binaries (`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`) are linked against a minimal runtime WITHOUT libm. Per-element scalar transcendentals are unavailable inside `__simt_vf__` functions. The official AscendC API catalog lists transcendentals in §9.1 "Math Library" (数学计算库) as SIMD APIs only (`LocalTensor<T>` + `sharedTmpBuffer` operands) — no public SIMT scalar equivalent.
- **Fix (verified in op#28 MultimodalRopePositionComputationWithGridBasedIndexing)**: Split the kernel into two launches:
  1. **SIMT kernel** materializes the transcendental's *input* into a dedicated GM buffer (e.g., `emb[total_tokens, head_dim]` fp32).
  2. **SIMD kernel** reads that GM buffer with `DataCopy` into `TQue<VECIN, 2>`, applies AscendC's `Cos()` / `Sin()` / `Exp()` / `Log()` high-level API on `LocalTensor<T>`, writes via `DataCopy` to `TQue<VECOUT, 2>`.

  Pair outputs (e.g. Cos + Sin on same input) in one SIMD pass to share MTE2 load:
  ```cpp
  LocalTensor<float> x = inQ_.DeQue<float>();
  LocalTensor<float> cosT = cosQ_.AllocTensor<float>();
  LocalTensor<float> sinT = sinQ_.AllocTensor<float>();
  LocalTensor<uint8_t> tmp = tmpBuf_.Get<uint8_t>();
  Cos<float, false>(cosT, x, tmp, count);
  PipeBarrier<PIPE_V>();
  Sin<float, false>(sinT, x, tmp, count);
  PipeBarrier<PIPE_V>();
  ```
  Cost: one extra GM round-trip (SIMT → GM → SIMD read-back). Usually cheap — Cos/Sin polynomial eval is compute-bound, not MTE2-bound.
- **Detection**: grep `ld.lld` output for `undefined symbol: <trig/exp/log>f`. Check if a `__simt_vf__` function references any of these in scalar form.
- **Prevention (Phase B checklist)**: For ops with per-element trig / exp / log (RoPE, sinusoidal PE, softmax with scalar `exp`, gaussian activation, softcap, etc.): plan SIMT + SIMD split in Phase A, do NOT attempt pure-SIMT.
- **Related**: Structural AIV constraint, not a bug — AscendC API catalog §9.1 already says transcendentals are SIMD-only. This entry formalizes the Phase A / Phase C takeaway for ops that miss the catalog lookup.
- **Evidence**: op#28 Phase C iter 2 (2026-04-22). First version used `__builtin_cosf(f)` / `__builtin_sinf(f)` in `__simt_vf__`; compile OK, link fail at `ld.lld`. Split into `mrope_build_emb_vf` (SIMT → fp32 GM) + `MropeCosSinApply` (SIMD `Cos<float>()` + `Sin<float>()`). Link OK, precision 50/50, perf 10.3x sum.
- **Status**: OPEN (structural). Not a candidate for CANN fix.
- **Distinct path — do not over-generalize "transcendentals don't link in AIV"** (added 2026-06-22): this entry is the AIV-VECTOR device-object link path (`KERNEL_TYPE_AIV_ONLY` → libm names undefined at `ld.lld`). The SEPARATE SIMT-SCALAR path DOES have working transcendentals: `expf`/`logf`/`log1pf` are declared `__simt_callee__` in `simt_api/math_functions.h` and link/run inside a VF callee (see OL-242). So "split SIMT+SIMD for transcendentals" (this EC, vector path) and "call the SIMT scalar intrinsic directly" (OL-242, scalar path) coexist — identify which path your callee is on before choosing. Cross-ref OL-242.

<!-- 迁移自 porter kb/target/ascendc/（EC-35，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
