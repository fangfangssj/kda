---
type: Runbook
title: "Quarantined candidate: use issue-window and register-readiness evidence to choose bounded unroll/prefetch"
description: "隔离候选（不得投送）：David V100/V130 源的 issue-window/寄存器就绪度假说——有界 unroll 暴露独立迭代、prefetch 缩短寄存器预留；A5 队列/寄存器容量与收益均未验证；OL-245 的 A5 实测边界仍为权威。"
tags:
- task.debugging
- topic.compilation
status: deprecated
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-WIKI-REGBASE-ISSUE-QUEUE"]
---

# CAND-WIKI-REGBASE-ISSUE-QUEUE

## 适用场景

- 适用平台与范围：none —— quarantined candidate，不声明任何生产目标适用性（卡内 applies_to 行）；validation_target 为 A5 / Ascend950PR，status=not_validated。
- 验证与未验证边界：见卡内 verified_on / unverified_on / source_scope 行（逐字节保留于正文）。
- 隔离状态：CANDIDATE / QUARANTINED FROM AUTOMATIC USE（卡内 status 行）——仅为反馈环保留，不得投送；canonical 既有卡为 optimizations/ol_245_simd_chains_regbase_vs_membase.md（OL-245；另自称 OL-54/OL-196 的源补充）。


> lifecycle: quarantined（KB_INDEX 标记；不得投送，仅为反馈环保留）

## CAND-WIKI-REGBASE-ISSUE-QUEUE: use issue-window and register-readiness evidence to choose bounded unroll/prefetch

`applies_to: none — quarantined candidate; no production target applicability is claimed`

`source_scope: architecture=DavidV100/V130; backend=RegBase/MicroAPI-source; A5=false`

`validation_target: A5 / Ascend950PR; status=not_validated`

`verified_on: exact sanitized V100/V130 source bytes and 8/8 referenced image bytes; no current-A5 device run`

`unverified_on: Ascend950PR queue/register capacities, dual-issue set, OOO policy, unroll factor, and performance benefit`

`status: CANDIDATE / QUARANTINED FROM AUTOMATIC USE — source supplement to OL-54/OL-196/OL-245, not NEW_VALIDATED_SOURCE`

- **Candidate hypothesis**: a long dependency chain can occupy the visible issue
  window so completely that later independent loop iterations never enter it. A
  bounded unroll can expose independent loads/ALU operations; load-prefetch followed
  by delayed consumers can shorten register-reservation time. Conversely, excessive
  unroll can increase live registers and issue backpressure. Treat this as a
  profile-directed scheduling hypothesis, not as "always unroll".
- **Scope fence**: the source explicitly describes David V100/V130 and quotes numeric
  queue/register capacities for that hardware. Those values and its dual-issue/OOO
  model are not imported as Ascend950PR facts. OL-245's measured A5 boundary remains
  authoritative: RegBase can lose when fine-grained calls do not amortize entry cost,
  and absence of RegBase never proves it is worthwhile.
- **A5 experiment**: on a current-A5 kernel whose timeline shows issue or
  register-readiness bubbles, compare baseline, bounded unroll, and
  unroll-plus-prefetch with identical math, memory traffic, tile, and call count.
  Record compiler output/register use, dependency-chain length, device time,
  precision, and determinism. Reduce the factor or reject the transform when spills,
  occupancy loss, instruction growth, or new hazards dominate.
- **Evidence binding**: sanitized source lines 2317-2565 and every referenced image
  are byte-bound in the
  batch-1 source manifest（A5Ops 绑定外部引用；本语料未收录目标）.
  Representative issue-window and code-order/execution-order diagrams were visually
  checked; they remain source evidence, not A5 measurements.
- **Promotion gate / falsifier**: require two independent Ascend950PR ops, exact
  source/binary/case identity, a measured issue/readiness bottleneck before the
  transform, and a repeatable gain without correctness/determinism regression. A5
  compiler scheduling that already exposes the iterations, or a controlled slowdown
  from register pressure, falsifies the recommendation for that scope.
