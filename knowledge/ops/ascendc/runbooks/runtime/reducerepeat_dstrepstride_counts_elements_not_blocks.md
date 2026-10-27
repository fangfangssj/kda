---
type: Runbook
title: "ReduceRepeat's dstRepStride counts elements, not 32B blocks"
description: "ReduceRepeat's dstRepStride counts elements, not 32B blocks"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-520"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-520 — OL-520: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC vector reduce; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 (4-29 FlashAttentionBwd rowdot, 2026-09-05)。

**OL-520** — **`ReduceRepeat`'s `dstRepStride` counts ELEMENTS, not 32B blocks** — the one stride parameter in the AscendC vector API that does not follow its neighbours, and inferring it from them silently corrupts every row except 0, 8, 16, …

- Former A5Ops ID: `OL-520`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC vector reduce; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 (4-29 FlashAttentionBwd rowdot, 2026-09-05)`

**Principle**: `ReduceRepeat<ReduceType::SUM, T, U>(dst, src, mask, repeatTime, dstRepStride,
srcBlkStride, srcRepStride, order)` writes ONE value per repeat and advances `dst` by
`dstRepStride` **elements**. In the same kernel, `BinaryRepeatParams` and `UnaryRepeatParams` repeat-stride fields are
**32B-block** granular. Do NOT generalise that to every field or every API — `DataCopyExtParams`
is direction-dependent (in the GM->UB form used here `blockLen` is in BYTES) — so check the
SPECIFIC field of the SPECIFIC API in the SPECIFIC direction rather than assuming a uniform unit. That
asymmetry is the trap: `dstRepStride = 1` yields `rc` CONTIGUOUS results, not results spaced 8
floats apart, and a write-back built on the block reading samples only rows 0, 8, 16, …

**Failure signature**: outputs wrong for most rows while a strided subset is correct. In the
observed case the bound profile reported `matched_ratio` ≈ 0.031 (threshold 0.9) on **all 50
cases and both outputs**. Use that severity as a TRIAGE ORDER, not as a ruling: a value that
low is a reason to check **layout/addressing first**, before the numerics you intended to change.
It does NOT establish that a numerical cause cannot reach such a value, and it does not exclude
numerics. (Had the strided read been *nearly* right and returned
~0.85, the natural misreading is "my numerical change was too aggressive", and a sound approach
gets abandoned for the wrong reason.)

**One observation on the order parameter**: switching the trailing
`ReduceOrder::ORDER_VALUE_INDEX` (default) to `ReduceOrder::ORDER_ONLY_VALUE` changed the built
DSO but produced **byte-identical results** for `ReduceType::SUM`. That is a single measurement on
one call shape — **not** a general finding that `ReduceOrder` is irrelevant, and not licence to
skip testing it in another context.

**Enum locations** (they are not under `compiler/tikcpp/`, which costs several searches):
`ReduceType {NONE=-1, SUM=0, MAX, MIN}` is in
`<CANN>/aarch64-linux/asc/impl/utils/common_types.h`; `ReduceOrder` is in
`<CANN>/aarch64-linux/asc/impl/basic_api/utils/kernel_utils_mode.h`.
