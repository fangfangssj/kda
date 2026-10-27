---
type: Runbook
title: "da Vinci VEC throughput tiers — Muls (Tier 0) is 8-16× faster than Div (Tier 4)"
description: "fp32 VEC throughput runs from Muls 8-16 elem/cyc (Tier 0) down to Div(vec,vec) 0.5-1 (Tier 4); count Tier-3/4 ops and PipeBarriers per row and minimize — normalize with Muls(x,1/rms), not Divs."
tags: [task.optimization, topic.precision, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.performance.analysis]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-257-davinci-vec-throughput-tiers.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
aliases: ["OL-257"]
---

# da Vinci VEC throughput tiers — Muls (Tier 0) is 8-16× faster than Div (Tier 4)

> 原 OKF v1 卡号：**OL-257**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - a SIMD kernel's per-row inner loop uses Divs/Div/Sqrt or emits >5 PipeBarrier<PIPE_V> per row for N≤256

## 现象 / 触发
A SIMD kernel is slower than expected and its per-row inner loop contains divide/sqrt ops or many pipeline barriers. da Vinci VEC instructions differ by up to ~16× in throughput, so the count of expensive ops per row directly bounds perf.

Approximate fp32 VEC throughput tiers:

| Tier | Operations | Approx elem/cycle | Per-row guidance |
|---|---|---|---|
| Tier 0 (fastest) | `Muls(vec, scalar)` | 8-16 | use wherever possible |
| Tier 1 (standard) | `Add`, `Mul`, `Sub`, `Cast` | 4-8 | normal vector work |
| Tier 2 (moderate) | `Abs`, `Max`, `Min`, `Relu` | 2-4 | acceptable |
| Tier 3 (expensive) | `Divs(vec, scalar)`, `Sqrt` | 1-2 | limit to ≤1 per row |
| Tier 4 (avoid) | `Div(vec, vec)` | 0.5-1 | avoid in perf paths |
| Fixed cost | `PipeBarrier<PIPE_V>` | ~10-50 cyc drain | target ≤5 per row for N≤256 |

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
Decision rule:
1. Count Tier-3/4 ops per row. If >1, look for conversion opportunities (see OL-256: `Divs(x,rms)` → `Muls(x,1/rms)` is precision-safe in fp32).
2. Count `PipeBarrier<PIPE_V>` per row. If >5 for N≤256, see OL-245 (regbase eliminates per-op barriers between chained VEC ops).
3. A normalization step (`x / rms`) should be `Muls(x, 1/rms)`, not `Divs(x, rms)`.

Concrete anchor:
```cpp
// EXPENSIVE (~400-800 cycles for N=256):
Divs(work, work, rms, N);         // Tier 3
PipeBarrier<PIPE_V>();
Div(work, work, scaleFp32, N);    // Tier 4

// CHEAP (~16-32 cycles for N=256):
float invRms = 1.0f / rms;
Muls(work, work, invRms, N);      // Tier 0
```

Evidence: add_rms_norm_quant V1→V2 A/B and selective_scan fwd perf-loop iters 1-4 (Ascend950PR_957b, 2026-06-23/24). Cross-refs: OL-256 (the Tier-0 substitute for Tier-3), OL-245 (regbase / barrier elimination), OL-258 (TQue double-buffering hides MTE latency).
