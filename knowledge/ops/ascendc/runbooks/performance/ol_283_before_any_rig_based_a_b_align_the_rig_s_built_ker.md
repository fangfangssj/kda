---
type: Runbook
title: "before ANY rig-based A/B, align the rig's BUILT kernel to the current production kernel (md5-verify) — a stage rig's built kernel can lag production by generations, making the A/B baseline wrong / the optimization inapplicable"
description: "Provenance: scan (cann_scan), selective_scan_fwd_simd R2 2026-07-24 — nearly measured an A/B on a rig base 2 generations behind production; intercepted before measurement."
tags: [task.optimization, topic.data_movement, topic.memory, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-283-before-any-rig-based-a-b-align-the-rig-s-built-ker.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: []
aliases: ["OL-283"]
---

# before ANY rig-based A/B, align the rig's BUILT kernel to the current production kernel (md5-verify) — a stage rig's built kernel can lag production by generations, making the A/B baseline wrong / the optimization inapplicable

> 原 OKF v1 卡号：**OL-283**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - you apply an optimization "modify construct X" to a stage rig's already-built kernel, but X does not exist in that kernel (it lags production) → nothing to modi

> Provenance: scan (cann_scan), `selective_scan_fwd_simd` R2 2026-07-24 — nearly measured an A/B on a rig base 2 generations behind production; intercepted before measurement.

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: any rig-based perf A/B (a5_ops ko/optimizer runs included); backend=ascendc`
`verified_on: selective_scan_fwd_simd — rig built kernel = pre-KO-2 ko1 (55d50c55, 0 Broadcast); production = 9199311c (KO-2, has the 3 UB Broadcasts). R2 = "replace those 3 Broadcasts with register Gather" — the 3-Broadcast target does not exist in the ko1 rig base → R2 inapplicable / A/B void`

**Symptom**: you apply an optimization "modify construct X" to a stage rig's already-built kernel, but X does not exist in that kernel (it lags production) → nothing to modify → the A/B is invalid / measures the wrong thing.

**Principle**: a stage rig's BUILT kernel is a snapshot that can be generations behind the current production kernel. An A/B against a stale rig base is a WRONG BASELINE — the optimization may target constructs the stale base doesn't have, or the "before" number isn't the real current-production number.

**Recommendation**: before any rig A/B, **align the production kernel into the rig + rebuild**, and gate the A/B on `rig_kernel_md5 == current_production_kernel_md5`. Bind the A/B baseline to the exact current production kernel md5, so a drifted base is caught before measurement rather than after. (Perf-A/B analog of the CLAUDE.md same-condition-A/B rule + OL-281's "provision the real thing, not a stale snapshot".)

**Cross-ref**: OL-282 (vehicle-identification — the OTHER rig-A/B trap); CLAUDE.md "性能对比必须是同条件 A/B". backend=ascendc.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-283（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
