---
type: Runbook
title: "A NO-GO / architecture-loses verdict from an un-optimized or broken prototype is not an architecture verdict — optimize BOTH to the FLOP roofline first"
description: "vec_ratio approx 0.99 saturates the vector-INSTRUCTION pipe, not the FLOP roofline; a NO-GO verdict on a broken or un-optimized prototype describes the prototype, not the architecture."
tags: [task.optimization, topic.precision, topic.implementation.synchronization, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.programming_model.execution_model.simt]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-231-unoptimized-prototype-not-architecture-verdict.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
aliases: ["OL-231"]
---

# A NO-GO / architecture-loses verdict from an un-optimized or broken prototype is not an architecture verdict — optimize BOTH to the FLOP roofline first

> 原 OKF v1 卡号：**OL-231**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - About to declare architecture X 'loses / NO-GO', or comparing an un-optimized X against an already-optimized Y, or calling a plateau because vec_ratio hit ~0.99

## 现象 / 触发
A comparison declares "architecture X loses / is NO-GO" (e.g. cooperative-SIMT prefix-scan vs vectorized SIMD), or a kernel is called plateaued because `vec_ratio ~= 0.99`. `applies_to: soc=Ascend950PR; cann=9.1.T500; op_class=all`.

Two recurring traps make a "loses" verdict an artifact rather than a real result:
1. **Broken / un-optimized prototype** — the verdict was measured on a degenerate prototype of X (scalar-bound preprocessing, wrong data layout, unfixed precision), not a real optimized X. "X is 40x / NO-GO" then describes the prototype.
2. **Asymmetric optimization** — X was compared against an ALREADY-optimized Y while X itself was still un-optimized. "Y beats X" is then a statement about effort, not architecture.

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
A "this architecture loses" verdict is only valid if that architecture was actually optimized. Before declaring a winner between two architectures, **fully optimize BOTH to their roofline, then compare.**

`vec_ratio ~= 0.99` means the vector-INSTRUCTION pipe is saturated, NOT that you are at the FLOP roofline — compute the FLOP-roofline gap before calling a plateau. vec ~= 0.99 is one of two cases needing opposite conclusions:
- **(a) lane-waste-bound** — narrow ops (e.g. N=16 on a 64-lane unit) under-fill the vector unit, so packing to full width cuts the op COUNT. The A5 vector unit is issue/latency-bound: a width-16, -32 and -64 op take identical time (measured microbench), so packing narrow->full is "free" per-op. A packing / N-widening execution model CAN help here.
- **(b) full-lane-compute-bound** — already using full lanes on real per-row FLOP. Irreducible by any layout trick; vec 0.99 IS the floor.

**Discriminator (run before chasing the gap):** msprof block-stub decomposition — is the lane-wasted slice actually DOMINANT, and would the packing model's machinery (`[g,n]<->[g,L]` layout conversion + per-row broadcast + scatter + the lost pipe-overlap from the barriers correctness needs) cost LESS than the lane-saving? If not, the roofline gap needs a DIFFERENT lever (cheaper per-row FLOP: faster transcendentals, cube offload), not packing.

## 证据
- selective_scan **backward** (2026-06-19): "cooperative-SIMT is the fastest architecture" held ONLY vs un-optimized SIMD. After the SIMD got batched-`Sum` vectorization (2.47x), SIMD >= coop (tie large / 2x small) — the winner flipped.
- selective_scan **forward** (2026-06-19): "SIMD is 40x-NO-GO" was a verdict on a broken regbase prototype (per-(l,n) scalar exp/softplus prestage, MERE ~= 45 / fp16 NaN). The build-ready natural-SIMD was precision-correct and merely un-optimized (vec_ratio 0.044); batched-`Sum` vectorization (vec 0.044 -> 0.975) made it beat the coop-SIMT baseline.
