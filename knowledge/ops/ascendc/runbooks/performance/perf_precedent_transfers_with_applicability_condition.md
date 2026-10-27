---
type: Runbook
title: "A perf precedent transfers only with its applicability condition — membase→regbase is governed by per-op element count (11.85x at 64 elem/op vs 1.25x at 2048)"
description: "A perf precedent transfers only with its applicability condition — membase→regbase is governed by per-op element count (11.85x at 64 elem/op vs 1.25x at 2048)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-327"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-327 — OL-327: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all (vector compute chains)。
- 验证来源：soc=Ascend950PR_9579; cann=9.1.0-beta.1。

A perf precedent transfers only with its applicability condition — membase→regbase is governed by PER-OP ELEMENT COUNT (measured 11.85x at 64 elem/op vs 1.25x at 2048); cross-refs OL-245

- Former A5Ops ID: `OL-327`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all (vector compute chains)`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0-beta.1`
`unverified_on: soc=Ascend910_9392 (arch 220x — the regbase API surface differs; re-measure before reuse)`

**Principle.** A speedup figure recorded on one operator is not a constant. It is the value of a
transformation *at that operator's operating point*, and quoting it without the axis that governs it
invites a reader to substitute it into a situation where it does not hold. **Record the discriminator, or
the number will be read as portable.**

**The measured instance.** [OL-245](../../optimizations/ol_245_simd_chains_regbase_vs_membase.md) records membase→regbase as ~15–33x on `selective_scan`. That is
correct, and it does not transfer by default. Same chain, identical total work (24 dependent steps over
2048 elements), oracle-verified, 4096 tiles / 56 blocks:

```
membase, 2048 elements per op :  153.07 us      <- large ops: the UB round-trip is already amortised
membase,   64 elements per op : 1446.17 us      <- 9.45x slower for the SAME total work
regbase (register-resident)   :  122.02 us
   regbase vs membase-2048 :  1.25x
   regbase vs membase-64   : 11.85x
```

**Decision rule (applies to ANY vector chain, not just this operator).** Measure the per-op element count
before deciding to convert. Near a full vector register per op, the round-trip regbase removes is already
small and the conversion buys little; at tens of elements per op the same conversion is worth an order of
magnitude. Two statements are therefore both wrong: *"regbase is worth 15–33x"* and *"regbase is only worth
1.25x on A5"*.

**Method note — the prediction came first.** The element-count hypothesis was written down with a falsifying
prediction ("drop to tens of elements and the advantage should rise sharply") *before* the 64-element arm was
built. A mechanism that merely explains an observed number after the fact is not established by it.

- **Evidence**: FlashKDA A5 port, 2026-08-01, `Ascend950PR_9579`, NPU exclusive. Both arms validated against
  a closed form (`r^24 - 1 = 0.023703`) and bit-identical to each other before timing.
- **Other instances (predicted)**: any elementwise chain where tile width is a tunable — normalisation
  epilogues, gate/decay folds, quantisation scale application. Also the inverse direction: a kernel already
  using wide ops has *already banked* this benefit and should not be scheduled for conversion.
