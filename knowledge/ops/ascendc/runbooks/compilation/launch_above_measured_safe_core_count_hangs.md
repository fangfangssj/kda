---
type: Runbook
title: "OBSERVED — a launch above a measured-safe core count **HANGS** rather than erroring; a hard-coded ceiling can exceed the measured-safe bound"
description: "OBSERVED — a launch above a measured-safe core count **HANGS** rather than erroring; a hard-coded ceiling can exceed the measured-safe bound"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["PB-77"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-77 — PB-77: OBSERVED — a launch above a measured-safe core count **HANGS** rather than erroring; a hard-coded ceiling can exc

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579；CANN 9.1.0.B150。
- 验证来源：a5_ops 4_GroupedMatmulSwigluQuant（2026-09-07），单 case 探针扫描 + 空闲卡上的成对 ABBA 窗口。

`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B150`
`verified_on: a5_ops 4_GroupedMatmulSwigluQuant, 2026-09-07, single-case probe sweep + a paired ABBA window on an idle card`
`mechanism: UNDETERMINED — only the boundary is measured`

Single-case probe (case 0), same binary, only `AOG_USED_CORE_NUM` varied, on an idle card:

    cores 20 / 24 / 26 / 28  -> returns in 0.0 s
    cores 30 / 32 / 36 / 40  -> HANG (timeout, no return, no error)

Confirmed by a paired ABBA window on the same binary, by the same author: `rc=0` at 24 cores,
`rc=124` at 40.

The kernel's Python wrapper carried `_MAX_CORES = 40` and, when `AOG_USED_CORE_NUM` was unset, fell
back to `min(_MAX_CORES, (m // 64 + experts) * n_blocks)`. Evaluated over the 60 frozen cases, that
expression requests ≥ 30 cores on **53 of 60** (computed from the formula; the hang itself was
measured on case 0 only), so the default path requests a hanging core count on most cases. It only
ever ran because the usual runner happened to pass an explicit 24.

⚠️ **Do NOT read this as "more cores than the physical count hangs".** The measured data contradicts
that: the platform config reports `cube_core_cnt = 32`, yet **30 and 32 also hang**. The platform
numbers are context, not an explanation.

**What is proven, and only this**: on this implementation and this host, ≤28 returns and ≥30 hangs;
a hard-coded 40 is unsafe as a default; 28 is the current measured-safe upper bound. The exact
boundary (29) was not probed, no threshold mechanism is established, and **28 must not be transferred
to another implementation or part without re-measuring**.

**Detection**: a hang, not a diagnostic. If arms time out while a plain `torch_npu` matmul on the same
card succeeds, suspect the launch width before suspecting card contention. On this operator the hang
was first misreported as external card contention.

**Candidate mechanism to test, not established**: [OL-226](../../optimizations/ol_226_whole_grid_syncall_blockdim_cap_equal_trip.md) — a
whole-grid `SyncAll` requires all blocks to be co-resident; this kernel uses `SyncAll<true>()`.
**Related**: the operator's par64 step raised the core-count cap from 20 to 40
(`min(20, m//128+e) -> min(40, m//64+e)`) together with `baseM` and `kRowChunk` 128 -> 64; its
performance gain is not attributable to any one of the three changes.
