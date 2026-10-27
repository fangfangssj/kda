---
type: Runbook
title: "In a two-arm control neither arm is exempt — get a value from outside both; a broken reference arm ran faster and framed the correct arm as wrong"
description: "In a two-arm control neither arm is exempt — get a value from outside both; a broken reference arm ran faster and framed the correct arm as wrong"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-329"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-329 — OL-329: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend950PR_9579; cann=9.1.0-beta.1。

In a two-arm control neither arm is exempt — get a value from outside both; a broken reference arm ran FASTER and framed the correct arm as wrong

- Former A5Ops ID: `OL-329`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0-beta.1`

**Principle.** Comparing a new implementation against a reference arm checks *consistency*, not
*correctness*. If the reference arm is wrong, the comparison reports the correct arm as the deviant one —
and in performance work the error is doubly seductive, because **an arm that computes less is also faster,
so "quicker" and "wrong" point the same direction.** Speed can never validate a probe.

**The measured instance.** A membase reference arm placed `DataCopy(out)` *before* its `PipeBarrier`, so the
MTE3 store raced the vector chain and shipped pre-compute data. It ran faster than the arm under test and
disagreed with it by 2% of full scale. Arm-vs-arm reasoning would have concluded the *new* arm was broken.
A closed form for the chain (`x <- x*r + c`, 24 times, from `x0 = 0`, so `x24 = r^24 - 1 = 0.023703`)
identified the reference as the faulty side in one run.

**Decision rule.** Every probe carries at least one value obtained from neither implementation — a closed
form, a CPU reference, or a control constructed to be equal on a dimension you are not measuring (for
example two arms moving identical byte counts so only the access pattern differs). Report the measured
error, not a tolerance band chosen after seeing it.

- **Evidence**: FlashKDA A5 port, 2026-08-01. After the fix both arms returned exactly `0.023703` and were
  bit-identical, and only then was the timing quoted.
- **Other instances (predicted)**: any A/B where the reference is "the existing production shape" — that
  framing is what licensed treating it as known-good.
