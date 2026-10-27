---
type: Runbook
title: "`basic_api` scalar-GM atomic RMW (`AtomicMax`/`AtomicAdd`/`AtomicMin`/`AtomicCas`/`AtomicExch`) is arch-gated OFF"
description: "`basic_api` scalar-GM atomic RMW (`AtomicMax`/`AtomicAdd`/`AtomicMin`/`AtomicCas`/`AtomicExch`) is arch-gated OFF"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["AtomicAdd", "AtomicCas", "AtomicExch", "AtomicMax"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-52 — PB-52: `basic_api` scalar-GM atomic RMW (`AtomicMax`/`AtomicAdd`/`AtomicMin`/`AtomicCas`/`AtomicExch`) is arch-gated OFF

> 关联：同一 PB-52 主题已由既有卡 [pb_52_phase_o5_re_measure_stages_the_flat_precision_eval.md](pb_52_phase_o5_re_measure_stages_the_flat_precision_eval.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend910_9382 (a3/V220)；CANN 9.0.0；api=AtomicMax/AtomicAdd/AtomicMin/AtomicCas/AtomicExch（basic_api scalar __gm__ RMW 形式）。
- 验证来源：soc=Ascend910_9382；CANN 9.0.0；arch=__NPU_ARCH__==2201。

`applies_to: soc=Ascend910_9382 (a3/V220); cann=9.0.0; bisheng=n/a; api=AtomicMax/AtomicAdd/AtomicMin/AtomicCas/AtomicExch (basic_api scalar __gm__ RMW form)`
`verified_on: soc=Ascend910_9382; cann=9.0.0; arch=__NPU_ARCH__==2201`
`unverified_on: whether a later CANN release adds 2201 support — re-grep the shipped header before relying on this being permanent`

The scalar global-memory atomic RMW intrinsics declared in `basic_api/kernel_operator_atomic_intf.h` (`AtomicMax<T>(__gm__ T*, T)` etc.) are wrapped `#if __NPU_ARCH__==5102||3101`, which **EXCLUDES** the a3/V220 target's actual `__NPU_ARCH__==2201` (confirmed by `basic_api/kernel_common.h`'s separate `==2201` branch). A kernel design assuming a device-side scalar atomic-max/add/etc. on a3/V220 will not compile.

- **Distinct from `SetAtomicAdd`**: `SetAtomicAdd<T>()` + `DataCopy(VECOUT→GM)` is a DIFFERENT mechanism (DMA-mode accumulate, widely used) and is NOT what this entry covers; the arch gate here is on the scalar `__gm__`-pointer RMW form only. (Max/Min DMA-mode variants unverified.)
- **Portable fallback**: when the problem reduces to a small-N sequential scan, launch with `blockDim=1` and use `GlobalTensor<T>::GetValue`/`SetValue` scalar GM accessors — these carry NO `__NPU_ARCH__` guard (unconditionally available) and single-core program order makes read-after-write race-free by construction (no atomics, no scratch-reset needed when Pass-1-writes and Pass-2-reads share the same launch + row set).
- **Grep-scope safety (folded-in near-miss)**: the "grep the shipped SDK header before assuming a primitive exists" discipline MUST be scoped to the public header tree (`$CANN_PATH/*-linux/asc/include/`), NEVER a raw `find $CANN_PATH -iname '*.h'` — on a real CANN install the latter also matches the FORBIDDEN op-impl source tree (`opp/built-in/op_impl/ai_core/tbe/impl/ops_*/ascendc/*`, off-limits for NPUKernelBench per CLAUDE.md "No CANN Source Code Copy"). Scope the grep to `asc/include/` up front. (NB: the worker note that surfaced this called the grep-before-assuming rule "OL-54" — that ID is actually "Reg-based SIMD"; the grep-first discipline lives in the spirit of OL-80/OL-84 "check the table/header before assuming", not OL-54.)
- **Status**: OPEN (arch-gated, not a bug per se — a real capability gap on 2201).
- **Evidence**: `12_KvRmsnormRopeCache` Part B (2026-07-16, a3 Ascend910_9382, CANN 9.0.0): pre-implementation grep found `AtomicMax` gated away from 2201 → avoided a wasted atomic-based dedup attempt (the exact failure mode where a prior fo/ko pair had assumed `Reg::Rsqrt` existed); used a `blockDim=1` sequential `GetValue`/`SetValue` last-writer-wins dedup instead → perf 0.5704×→0.9329×.
- **Cross-reference**: OL-80 / OL-84 (check the shipped header/table before assuming an API compiles); CLAUDE.md "No CANN Source Code Copy — NPUKernelBench ONLY" (the grep-scope guard).

---