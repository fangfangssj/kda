---
type: Runbook
title: "SIMD elementwise kernel segfault (exit 139) with blockDim > 1 on A5 — multi-core-first diagnostic, NBLK=1 as fallback"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=elementwise,fused-elementwise"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-78-simd-elementwise-kernel-segfault-exit-139-with-blo.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: ["950"]
aliases: ["EC-78"]
---

# SIMD elementwise kernel segfault (exit 139) with blockDim > 1 on A5 — multi-core-first diagnostic, NBLK=1 as fallback

> 原 OKF v1 卡号：**EC-78**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - SIMD elementwise kernel launched with blockDim=56 (data-parallel — all cores run identical code on identical data) crashes with exit 139. Distinct from a kernel

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=elementwise,fused-elementwise`
`verified_on: soc=Ascend950PR_957b; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A2/A3 — elementwise SIMD multi-core may work on A3; A5-specific trigger unconfirmed)`
`UPDATED 2026-06-24`: Originally observed as "NBLK=1 required" (swi_glu V1). Revised to "multi-core-first diagnostic" after V5 proved GetBlockIdx() outer_blocks partitioning works correctly and delivers 2.11× geo_mean improvement.

- **Symptom**: SIMD elementwise kernel launched with `blockDim=56` (data-parallel — all cores run identical code on identical data) crashes with exit 139. **Distinct from** a kernel that uses `GetBlockIdx()` to partition independent outer_blocks across cores — that pattern works correctly (see OL-254).
- **Root cause**: The crashing pattern uses `blockDim=N` data-parallel launch (all N cores run the SAME tile loop on the SAME data — no per-core work partition). This triggers a launch-time crash on A5 CANN 9.0.0. The crash is specific to data-parallel (identical-work) multi-core launch, NOT to multi-core in general.
- **Fix (diagnostic)**:
  1. FIRST verify the kernel uses GetBlockIdx() to partition work (see OL-254). If NOT — add per-core partitioning. This is the preferred fix and enables multi-core speedup.
  2. If the kernel ALREADY uses GetBlockIdx() partitioning AND still crashes → set `NBLK=1` as a diagnostic probe. If NBLK=1 fixes it, the bug is in the partition logic.
  3. NBLK=1 as a PERMANENT choice ONLY when outer_blocks = 1 (physically cannot partition).
- **Detection signal**: kernel compiles clean, pybind launches, process exits 139 immediately (before any kernel-side computation). If the crash is at launch time (not during kernel execution), suspect this.
- **Evidence**:
  - swi_glu V1 (2026-06-23, Ascend950PR_957b CANN 9.0.0): data-parallel blockDim=56 → segfault. NBLK=1 resolved → 50/50 bit-exact PASS, but perf 0.72× geo_mean with 0.07-0.16× on large shapes.
  - swi_glu V5 (2026-06-24, same hardware): GetBlockIdx() outer_blocks partitioning with nblk=32 → 50/50 bit-exact PASS, perf 1.52× geo_mean (2.11× vs V1), 41/50 faster-than-ref. Large-shape geo_mean 1.35× (vs V1 0.16× = 8.5× improvement). Proves multi-core elementwise IS correct and performant when using per-core work partition.
- **Cross-ref**: EC-17 (nblk=1 for sub-alignment chunk overwrite — different root cause), OL-214 (single-core-first testing methodology — NBLK=1 as diagnostic probe), OL-254 (multi-core outer_blocks partitioning — the preferred pattern), P-P114 (multi-core outer_blocks template).

<!-- 迁移自 porter kb/target/ascendc/（EC-78，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
