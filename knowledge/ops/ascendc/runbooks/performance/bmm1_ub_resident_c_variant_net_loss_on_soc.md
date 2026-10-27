---
type: Runbook
title: "The tested BMM1 UB-resident-C variant is a large net loss on this SoC, even though the API supports it"
description: "The tested BMM1 UB-resident-C variant is a large net loss on this SoC, even though the API supports it"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-524"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-524 — OL-524: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC MIX matmul (KFC client); op_class=fused attention。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 (4-54 OptimizedFlashAttention, 2026-09-05)。

**OL-524** — **the tested BMM1 UB-resident-C variant is a large net LOSS on this SoC, even though the API supports it**

- Former A5Ops ID: `OL-524`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC MIX matmul (KFC client); op_class=fused attention`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 (4-54 OptimizedFlashAttention, 2026-09-05)`

**Counter-example, recorded so THIS VARIANT is not re-attempted blind** (BMM1's C only, `BLOCK_N=256`, fp32; BMM2's C was left on GM and is untested). `IterateAll(const LocalTensor<DstT>&)` **is**
implemented on the arch35/v310 path (内部路径（不公开）mx_matmul_impl.h:174), so landing the matmul result
directly in UB is a supported library call, not hand-rolled cross-core sync. Changing BMM1's C from
`TPosition::GM` to `TPosition::VECCALC` removes a full S-tile round trip through global memory
(64KB out + 64KB back per kv iteration at `BLOCK_N = 256`).

It **builds clean, runs, and is numerically correct** — `P 50/50 PASS_T1`, no hang. It is also **substantially slower on the large cases measured** — the per-case ratios below fall in a
band, they are not a uniform 28%:

```
geomean_speedup  0.4886 (UB-resident C)  vs  0.6271 (C through GM), same denominator
case 40 0.548 -> 0.393   case 43 0.362 -> 0.268   case 47 0.358 -> 0.265
small shapes stay near 1.0 (case 2 = 1.257); the penalty GROWS with problem size
```

**Scope — what this does and does not establish.** It establishes that **this variant, as configured,**
is a net loss. The two geomeans come from **different runs** (each internally paired against its own
reference leg), with **one observation per configuration** and no variance estimate. It does **not**
establish that the GM round trip is cheap: the change also altered the matmul's internal
tiling and cube<->vector handshake, unisolated, so a real traffic saving may have been swamped.
Mechanism is **open**. Recorded as a counter-example, not as a verdict on where the time goes.

**Adjacent, from the headers**: eliminating the *other* round trip (P back to GM for BMM2's A) is not
a UB question — 内部路径（不公开）matmul_client.h:292 routes a LocalTensor A operand through `TPosition::TSCM`,
i.e. an **L1** allocation with its own budget.
