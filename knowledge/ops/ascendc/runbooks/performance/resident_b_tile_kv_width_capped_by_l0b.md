---
type: Runbook
title: "For a resident (non-streamed) B tile, the kv-tile width is capped by L0B, and the cap is computable before spending a device window"
description: "For a resident (non-streamed) B tile, the kv-tile width is capped by L0B, and the cap is computable before spending a device window"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-522"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-522 — OL-522: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC cube/MIX matmul; op_class=fused attention / any tiled GEMM。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 (4-54 OptimizedFlashAttention, 2026-09-06)。

**OL-522** — **for a RESIDENT (non-streamed) B tile, the kv-tile width is capped by L0B, and the cap is computable before you spend a device window**

- Former A5Ops ID: `OL-522`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC cube/MIX matmul; op_class=fused attention / any tiled GEMM`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 (4-54 OptimizedFlashAttention, 2026-09-06)`

**Rule**: where the B operand tile is held **resident** in L0B, it occupies
`baseK * baseN * sizeof(T)` bytes. On this SoC `l0_b_size = 65536`
(`Ascend950DT_9582.ini`). **For the tested configuration — fp32, `baseK = 64`, this operator's
resident scheme** — that pins `baseN = 256` at exactly 65536 bytes, **100% of L0B**; `baseN = 512`
needs 200% and cannot run.

**Scope of the negative.** The 512 result rules out **that configuration**. It does **not** show
that a 512-wide kv tile is impossible in general: a smaller `baseK`, a K-sharded or N-sharded
schedule, or a streamed/multi-pass B feed all change the resident footprint and were **not tested
here**.

**Why this is worth writing down**: raising the kv-tile width is one of the highest-value knobs on a
flash-attention kernel — it divides `kvLoops`, and BMM2 emits `[BLOCK_M, D]` **per iteration**
regardless of N, so halving `kvLoops` halves BMM2's entire output volume. Going 128 -> 256 was worth
**+19%** on this operator. The natural next step, 256 -> 512, is **not a tuning question** — it is
arithmetically impossible, and the failure is **not** a clear "out of memory": it is a runtime
`507015` "device error type 3", raised **only on the cases whose `validN` actually reaches 512**.

**Measured boundary matches the arithmetic case for case** (same build, one run each):

```
Skv <= 256  ->  PASS, output sums bit-identical to the 256 build   (cases 8, 11, 12-18)
             (an AGGREGATE check -- equal sums do not prove element-wise equality)
Skv >= 512  ->  FAIL 507015                                        (cases 20, 30, 47)
```

Small shapes passing is what makes this expensive to misdiagnose: a spot check on a few small cases
looks like the change works.

**Check before building**: `baseK * baseN * sizeof(T) <= l0_b_size`, and separately
`baseM * baseK * sizeof(T) <= l0_a_size` and `baseM * baseN * 4 <= l0_c_size`. Read the sizes from
`<CANN>/aarch64-linux/data/platform_config/<SOC>.ini`. When one is at exactly 100%, that axis is
**closed by capacity, not by search** — a stronger statement than a sweep that found a plateau,
because a plateau always leaves "maybe there is a value I did not try".
