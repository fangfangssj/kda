---
type: Runbook
title: "Vectorising a compensated row reduction without giving up exactness"
description: "Vectorising a compensated row reduction without giving up exactness"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-521"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-521 — OL-521: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC vector; bisheng=n/a; op_class=reduction / compensated-summation。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 (4-29 FlashAttentionBwd rowdot, 2026-09-05)。

**OL-521** — **vectorising a compensated row reduction WITHOUT giving up exactness**: extract each lane position into its own contiguous array with a single-bit `ReduceRepeat` mask, then run the TwoSum tree on aligned vectors — do not halve in place, and do not drop the residual

- Former A5Ops ID: `OL-521`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC vector; bisheng=n/a; op_class=reduction / compensated-summation`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 (4-29 FlashAttentionBwd rowdot, 2026-09-05)`

**Problem**: a per-row compensated reduction (Veltkamp/Dekker `TwoProduct` + pairwise `TwoSum`
tree + scalar Neumaier tail) is correct but expensive — measured at **13.5x above the stage's own
memory floor**, with the scalar tail alone 37.2% of the kernel. The obvious fixes both fail:

1. **Halving in place below 8 lanes is impossible.** Vector operands need 32B alignment; the
   tree's `base + h` is aligned only while `h` is a multiple of 8, so 8→4→2→1 is blocked. This is
   the same wall behind two recorded `507035 "vector core exception"` faults in this operator.
2. **Batching the tree across rows with `BinaryRepeatParams` repeat strides REGRESSED 1.39x**
   (58,187 → 80,991 us) while passing precision 50/50 — the transformation was numerically
   neutral, so the gate could not see that it was a bad idea. NOTE the scope: that measures the
   NET of fewer instruction issues MINUS new stride/scheduling cost; it does **not** prove the
   issue-overhead term is zero.

**Technique that works** — stated for **fp32** operands, the **AscendC vector API used here**, and
**this operator's scratch layout**; other dtypes, APIs or layouts are not covered by this
measurement. `ReduceRepeat` accepts a `uint64_t mask[]`, so a **single-bit mask** extracts lane *j*
of every row into its own CONTIGUOUS `[rc]` array while the source base stays at the aligned
row-major buffer. With `nTail` such arrays, the compensated `TwoSum` tree runs across them with
plain aligned vector ops — **identical operations, identical pairing, identical per-element order,
exact residual preserved**. In this arrangement no operand base is formed by halving in place, so
the sub-32B alignment constraint is not encountered (that is a property of THIS layout, not a
general impossibility claim about halving). Scratch reused buffers that were dead after
`TwoProduct` **in this kernel**, hence no additional UB **here** — not a general guarantee.
Likewise the per-row residual `ReduceSum` becomes chunked `ReduceRepeat` over 64 fp32 lanes; with
fp32 each chunk base at `c*64` floats is `c*8` 32B blocks and therefore aligned **for this element
size** — recompute the arithmetic for any other dtype.

### Evidence locators

| item | locator |
|---|---|
| candidate DSO (both P and F legs of the D1 run) | sha256 `2daf5b9fe86cffea7cbc1308b441b4161581ede31b87f63270989c1987304b16` |
| **D1 run** — P 50/50 + F geomean **1.051381671565124**, `raw_asc=50` + `raw_ref=50`, in-process binding | `Ascend950DT host:/root/triton-op29t2-20260905t0155z/ACCEPT29_D1_20260905T1223Z` |
| earlier run — F geomean **1.0264** (card 0) | `Ascend950DT host:/root/triton-op29t2-20260905t0155z/EVIDENCE_op29_doublepass` |
| `rowdot` component probe — candidate arm | `…/prof_hwexp` |
| `rowdot` component probe — per-row `ReduceSum` removed | `…/prof_norsB` |
| `rowdot` component probe — scalar Neumaier tail removed | `…/prof_notail` |

⚠️ **The 1.0264 run and the D1 1.0514 run are DIFFERENT runs on different cards.** They are listed
separately and must not be presented as one measurement or paired against each other; each keeps
its own provenance. The 2.17x `rowdot` figure quoted above comes from the probe directories in the
same table, not from either geomean run.

**Do NOT simply drop the compensation to buy speed.** Measured on the same operator: dropping the
tail residual reached the perf gate (geomean 1.0178) but scored **49/50**, failing
`[1,300,56,128]` fp32 at `matched_ratio` 0.8687. The compensation is load-bearing — marginally,
and only on the largest fp32 shape, which is exactly where a small-sample check would miss it.

**Result**: `rowdot` 58,187.2 → 26,783.3 us (**2.17x**) with precision **50/50 PASS_T1**
maintained, taking the operator from geomean 0.8826 to **1.0264** against a 1.0 gate. Untouched
kernels reproduced to within 0.07% across the arms — recorded as an OBSERVATION that no gross
measurement drift occurred. It is not a significance test and does not by itself establish
causation for the delta.
