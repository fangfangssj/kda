---
type: Runbook
title: "msprof Task Wait Time is skewed — quote quantiles not the mean, and split by task position before attributing"
description: "msprof Task Wait Time is skewed — quote quantiles not the mean, and split by task position before attributing"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-332"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-332 — OL-332: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all (multi-launch kernels)。
- 验证来源：soc=Ascend950PR_9579; cann=9.1.0-beta.1。

msprof Task Wait Time is skewed — quote quantiles not the mean, and split by task POSITION before attributing

- Former A5Ops ID: `OL-332`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all (multi-launch kernels)`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0-beta.1`

**Principle.** Summing `Task Wait Time` over a run and dividing by total duration produces a number that
looks like "how much time we spend waiting" and is not that. The distribution is dominated by a handful of
outliers, so **quote quantiles, and split by task position before attributing the wait to anything.**

```
observed over 300 tasks:  min 0 · p25 0 · median 0.027 · p75 0.125 · p90 1.059 · p99 332.8 · max 572.7 us
naive aggregate  sum(wait)/sum(duration) = 26.9%     <- reads as "the handshake is expensive"
dominant op only median 0.016 us on a 28.99 us task  = 0.06% ; 100/240 tasks wait EXACTLY 0
```

**The discriminator is position, not magnitude.** The ten waits above 50 us sat at task indices 2, 4, then
20, 40, 60 … 160 — i.e. every 20 tasks, which was exactly one iteration, and always the iteration's *first*
kernel. So they are per-iteration head latency (launch chain / previous-iteration drain), not cross-core
handshake cost. Indices 2 and 4 were genuine one-off warm-up. **In-kernel handshake cost and launch-chain
cost are indistinguishable in the total and separable only by position.**

- **Evidence**: FlashKDA A5 port, 2026-08-01. The aggregate would have justified redesigning a handshake
  that measurably costs 0.1%; cf. [OL-313](../runtime/handrolled_aic_aiv_handshake_arch_scoped_flag_and_mode.md) for the correctness contract of that same handshake, which is
  a separate question from its cost.
- **Other instances (predicted)**: any operator launched once per chunk/tile in a host loop — the first
  launch of each iteration pays the drain, and with enough iterations that dominates the wait column.
