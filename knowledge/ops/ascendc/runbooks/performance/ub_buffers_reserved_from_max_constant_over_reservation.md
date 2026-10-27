---
type: Runbook
title: "UB buffers reserved from a MAX_ constant instead of the value actually used: a recurring, compounding over-reservation"
description: "UB buffers reserved from a MAX_ constant instead of the value actually used: a recurring, compounding over-reservation"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-523"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-523 — OL-523: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC TPipe/InitBuffer; op_class=any。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 (4-54 OptimizedFlashAttention, 2026-09-05/06)。

**OL-523** — **UB buffers reserved from a `MAX_` constant instead of the value actually used: a recurring, compounding over-reservation**

- Former A5Ops ID: `OL-523`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC TPipe/InitBuffer; op_class=any`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 (4-54 OptimizedFlashAttention, 2026-09-05/06)`

**Pattern**: `pipe->InitBuffer(buf_, <MAX- or tile-shaped expression>)` where the tensor is only ever
used with a much smaller count. Three instances were found in **one 508-line kernel**, together
holding ~25% of the 248KB UB hostage:

| buffer | reserved | actually used | factor |
|---|---|---|---|
| `biasBuf_` | `BLOCK_M * BLOCK_N * 4` = 32768 B | one `BLOCK_N` row = 512 B | **64x** |
| `accBuf_` | `BLOCK_M * MAX_D * 4`, `MAX_D=128` | `d_k = 64` | **2x** |
| `oPartBuf_` | `BLOCK_M * MAX_D * 4`, `MAX_D=128` | `d_k = 64` | **2x** |

**Why it matters beyond tidiness**: UB is the binding constraint on tile size, and tile size is where
the performance is. **The over-reservation was not costing memory, it was costing the tile.**

**Attribution is to the configuration combination, not to the buffer recovery alone.** The +19%
(`BLOCK_N` 128 -> 256) and the further +13% (`BLOCK_M` 64 -> 128) were each measured on a **bundle**.
For `BLOCK_M = 128` the bundle was: recovering the `MAX_D` over-reservation, aliasing a dead buffer,
**and halving the softmax workspace cap 32768 -> 16384** — that third item is a coupled enabler
without which the tile does not fit (250.0KB -> 234.0KB of 248.0KB), and its own cost was assessed
only by a single-observation proxy at a *different* tile. None of the three is individually
credited.

**Detection** (cheap, static): for each `InitBuffer`, compare the reserved byte count against the
largest `count` argument ever passed to an op on the tensor from that buffer's `Get<T>()`. Any gap
beyond ~2x is worth reading. Do not trust a `MAX_` name to mean "and it is used".

**Safety note when tightening a `MAX_`** — and read this narrowly. Lowering such a constant
**narrows the supported input domain**, so it is only appropriate where the **input contract already
excludes** the wider range (here: every case in the frozen set is `d_k = 64`). A pre-existing
`TORCH_CHECK(D <= FA54_MAX_D)` guarantees only that an out-of-range input is **explicitly rejected
rather than silently under-allocated** — it does **not** make the narrowing itself reasonable, and it
is not a substitute for the contract. Tightening a bound with no such guard is more dangerous again.

**Related, same operator**: a buffer whose contents are dead after a consuming call can be aliased
instead of reserved twice — here `p` (softmax output) reuses `s` (softmax input), because
`SoftmaxFlashV2`'s fp32 path reduces the full `src` row to the row max before writing `dst`
(内部路径（不公开）softmax_flashv2_update_impl.h:118 vs `:138`) and already performs `Exp(dst, dst)` in place. That
was confirmed by a **precision leg of its own** — 50/50 PASS_T1 with aliasing as the only change —
not by reading the implementation alone.
