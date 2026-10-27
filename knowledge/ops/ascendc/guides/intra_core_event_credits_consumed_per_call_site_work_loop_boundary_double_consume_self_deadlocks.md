---
type: Guide
title: "Intra-core event credits consumed per call SITE — work-loop boundary double-consume self-deadlocks"
description: "One credit across an item boundary + two wait call sites (item-head + first-tile) = first tile of item 1 waits on empty counter (matching Set later in same thread) -> deterministic 507014. Fix: reset the any-issued flag after the item-head wait. Count Sets vs Waits across the loop boundary. Found on 11_BigBirdAttention eMte1Mte2_."
tags: [task.debugging, topic.implementation.synchronization, topic.implementation.pipeline]
status: draft
sources: []
created_at: "2026-09-26T12:23:46Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases:
- OL-613
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# Intra-core event credits consumed per call SITE — work-loop boundary double-consume self-deadlocks

## Evidence

- **run_id**: 5450b083-4f53-4620-a1b2-8656e7bb9889
- **soc**: Ascend950DT_9582
- **clock**: 2026-09-26T12:23Z
- **trigger**: 11_BigBirdAttention: X-matrix discriminator (1-item DONE / 2-item WEDGED) + event accounting audit

## 现象 / 触发
MIX kernel loops work items per core. An intra-core event (MTE1_MTE2) is Set once per
tile (after L1->L0 LoadData) and waited at TWO call sites: the item head ("previous
item's loads retired") and the first in-item tile ("previous tile's loads retired").
Exactly one credit crosses the item boundary; the item-head wait consumes it; the
first-tile wait then blocks on an empty counter whose matching Set issues later in the
SAME thread -> deterministic self-deadlock at item 1 (507014). Single-item and 1-item/
core probes cannot see it.

## 修法 / 规则
After the item-head wait, reset the "any producer issued" flag (e.g.
`anyLoadData_ = false`) so the first in-item tile re-establishes the "first skips"
invariant of item 0. Audit rule: for every SetFlag/WaitFlag event, count Sets vs Waits
across the work-loop boundary, not just within one iteration; every wait must have a
pending Set, and pending must stay <= the hardware depth (AIV V-pipe events wedge at
depth 2).
