---
type: Runbook
title: "Oversubscription helps unsorted bwd (-28%) but hurts sorted bwd (+17%)"
description: "Block oversubscription (nblk=448 vs 56) reduced unsorted backward time by 28% by dispersing atomicAdd contention across more time-sliced blocks. But after implementing sorted-edge register accumulatio"
tags: [task.optimization, topic.implementation.atomic, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-8-oversubscription-helps-unsorted-bwd-28-but-hurts-s.md
    role: primary
created_at: "2026-07-10T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-8"]
---

# Oversubscription helps unsorted bwd (-28%) but hurts sorted bwd (+17%)

> 原 OKF v1 卡号：**OL-8**。
> 现象分类：perf_regression；观测置信度：inferred（推断）。
> 触发信号：
> - when tuning launch parameters (nblk) for scatter-add backward kernels

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为推断沉淀（inferred），按卡内声明范围消费。

## 教训 / 根因
Block oversubscription (nblk=448 vs 56) reduced unsorted backward time by 28% by dispersing atomicAdd contention across more time-sliced blocks. But after implementing sorted-edge register accumulation (which eliminates atomicAdd contention entirely), the same oversubscription increased backward time by ~17% due to pure launch overhead with no contention left to disperse. The root cause is that oversubscription's benefit comes solely from dispersing atomicAdd contention -- once sort+register-accum removes that contention, only the overhead remains. Launch parameter sweeps must be re-run on the final kernel variant, not carried over from intermediate versions.

## 证据
EXPERT_FEEDBACK.md E7-8 (nblk sweep tables for unsorted vs sorted), MSPROF_AGENT_GUIDE.md (nblk=56 vs 448 msprof data)

<!-- LLM-辅助分类迁移(convert 二次 pass, migrate_ol_dispositions.py)。confidence=inferred,待人工确认。原 OL-8。 -->
