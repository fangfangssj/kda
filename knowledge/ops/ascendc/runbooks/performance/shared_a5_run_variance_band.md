---
type: Runbook
title: "Shared-A5 run variance for us-scale kernels is ±15% of mean ratio — report the band"
description: "Shared-A5 run variance for us-scale kernels is ±15% of mean ratio — report the band"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-SHARED-A5-RUN-VARIANCE-BAND"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-SHARED-A5-RUN-VARIANCE-BAND — Shared-A5 run variance for us-scale kernels is ±15% of mean ratio — report the band

## 适用场景

- 适用平台与范围：共享设备的性能测量；op_class=小 case 占比高的 benchmark。
- 验证来源：13_Cat M-I run-1 ko-2 L5（2026-09-17 外部作业争用窗口观测）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: shared-device perf measurement; op_class=tiny-case-heavy benchmarks`
`verified_on: 13_Cat M-I run-1 ko-2 L5 (foreign-job contention window 2026-09-17)`

Identical bytes measured 0.93x and 0.79x in two runs under a shared device; small-case device time 3us vs 9.4us across runs with large-case ratios stable (0.87-0.91x). Any perf claim on tiny-case-heavy benchmarks needs the variance band reported, not a point number; contention-window numbers never enter conclusions.

- **Cross-ref**: OL-95 (large-case wall-clock re-measure lineage). backend=ascendc.
