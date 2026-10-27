---
type: Runbook
title: "Select with SELMODE::VSEL_CMPMASK_SPR applies the compare mask with PERIOD 64 elements for count > 64 on dav-m"
description: "dav-m200 上 Select(SELMODE::VSEL_CMPMASK_SPR) 的掩码按 64 元素周期重复——count>64 时 64 位后的比较结果错位"
tags: [task.debugging, topic.compilation, topic.implementation.masking]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-112"]
---

# Select with SELMODE::VSEL_CMPMASK_SPR applies the compare mask with PERIOD 64 elements for count > 64 on dav-m

**PB-112** — `Select` with `SELMODE::VSEL_CMPMASK_SPR` applies the compare mask with PERIOD 64 elements for count > 64 on dav-m200 (Level-2 count impl does `set_cmpmask` once + single `vsel` under `SetMaskCount`; the cmpmask SPR does NOT advance per 64-element repeat) — any `Compares`+`Select` pair at tile counts > 64 silently computes garbage. Proven by on-device stage-dump bisect (period-64 simulation match=1.0000). Mitigation: mask-free arithmetic chain (CAND-M200-CANCELLATION-FREE-ARITHMETIC-SELECT) or Gather-select (CAND-DAV-M200-GATHER-SELECT); never stop at "platform limitation" — bisect. gelu 310p 2026-08-12 (29/29 T1, det 29/29). Cross-ref [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md)/[`PB-108`](pb_108_duplicate_uint8_t_is_rejected_on_dav_m200_sdk_dt.md)/[`PB-113`](pb_113_shiftleft_shiftright_are_not_supported_on_dav_m2.md), [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md). [dav-m200/Ascend310P3, vector-compute, porting]

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a (SDK impl defect); op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (gelu 310p, 2026-08-12 — on-device stage-dump bisect)`
`unverified_on: soc=Ascend310B, Ascend910, Ascend950PR (cmpmask SPR advance is a per-SoC impl detail — other SoCs' Select paths are separate code; OL-105's ShiftLeft/Select suspicions were 351x-side hypotheses, not this)`

- **Severity**: HIGH — compiles clean, runs without error, and silently produces wrong values for every element i ≥ 64 of a Compare/Select tile.
- **Symptom**: a `Compares`/`Select` pair (Level-2 count form) produces outputs as if the compare mask REPEATED every 64 elements — element i≥64 is selected by mask bit i%64.
- **Mechanism**: the dav-m200 Level-2 count impl (`VselImpl` in CANN 安装树内 impl/dav_m200/kernel_operator_vec_cmpsel_impl.h) does `set_cmpmask(sel)` ONCE, then a single `vsel(..., repeatTime=1, ...)` under `SetMaskCount(0, count)`. On device the cmpmask SPR does NOT advance per 64-element repeat, so the mask applies with period 64. Verified by stage-dump bisect (13 stages, one extra build cycle): a mask-reuse-period-64 simulation matched the observed outputs with match=1.0000.
- **Mitigation (preference order)**: (1) eliminate Compare/Select on dav-m200 entirely — mask-free arithmetic chain (clamp tail-kill + arithmetic sign blend; see CAND-M200-CANCELLATION-FREE-ARITHMETIC-SELECT) or Gather-based select (CAND-DAV-M200-GATHER-SELECT); (2) if a Select is unavoidable, keep count ≤ 64 per call.
- **Triage note**: do NOT declare "platform limitation" and stop at the first Select-based failure — the period-64 signature is provable on-device with a stage-dump bisect (dump intermediates per stage, simulate candidate mask semantics against the dumps) in one extra build cycle.
- **Evidence**: gelu 310p (2026-08-12, port_a3_to_a5 TARGET=310p) — failures_ledger precision iter 2: max_abs_diff_large from period-64 mask reuse; root-caused by the on-device stage-dump bisect (period-64 simulation match=1.0000), fixed by deleting Compare/Select entirely → 29/29 Tier-1 PASS, det 29/29 bit-identical. Kernel anchor `代码片段 gelu/gelu_kernel.h.snippet.h（见文末）` (workspace scratch, gitignored).
- **Other instances (predicted)**: any dav-m200 kernel using `Compares`/`CompareScalar` + `Select` at tile counts > 64 — sign blends, clamp-via-select, predicate masks (VSEL_CMPMASK_SPR is the default cmpmask path). Cross-ref [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md)/[`PB-108`](pb_108_duplicate_uint8_t_is_rejected_on_dav_m200_sdk_dt.md)/[`PB-113`](pb_113_shiftleft_shiftright_are_not_supported_on_dav_m2.md) (the dav-m200 missing/broken-API class — grep the per-SoC impl header per [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) before using a new API), CAND-DAV-M200-GATHER-SELECT, CAND-M200-CANCELLATION-FREE-ARITHMETIC-SELECT.

## 内嵌代码片段

### gelu/gelu_kernel.h.snippet.h

```cpp
            uint32_t chunks = chunksLeft > TILE_CHUNKS ? TILE_CHUNKS : static_cast<uint32_t>(chunksLeft);
            uint32_t num = chunks * CHUNK_ELEMS;
            CopyIn(offset, num);
            Compute(num);
            CopyOut(offset, num);
            offset += num;
            chunksLeft -= chunks;
        // R(z): Horner in u = min(z, ZMAX)*U_SCALE - 1 (domain-clamped so the
        // poly never extrapolates: R(ZMAX) is bounded and f below kills the tail)
        Mins(tC, tA, ZMAX_F, num);
        Muls(tC, tC, U_SCALE, num);
        Adds(tC, tC, -1.0f, num);                        // u
        Duplicate(tD, R_COEF[18], num);
        for (int k = 17; k >= 0; --k) {
```

## 补充证据（2026-10-10 合并自 `dav_m200_vsel_cmpmask_spr_period_64.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## PB-112 — PB-112: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，__CCE_AICORE__=200）；CANN 9.0.0；op_class=all（SDK 实现缺陷）。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（gelu 310p，2026-08-12 — on-device 分阶段转储二分）。

**PB-112** — `Select` with `SELMODE::VSEL_CMPMASK_SPR` applies the compare mask with PERIOD 64 elements for count > 64 on dav-m200 (Level-2 count impl does `set_cmpmask` once + single `vsel` under `SetMaskCount`; the cmpmask SPR does NOT advance per 64-element repeat) — any `Compares`+`Select` pair at tile counts > 64 silently computes garbage. Proven by on-device stage-dump bisect (period-64 simulation match=1.0000). Mitigation: mask-free arithmetic chain (CAND-M200-CANCELLATION-FREE-ARITHMETIC-SELECT) or Gather-select (CAND-DAV-M200-GATHER-SELECT); never stop at "platform limitation" — bisect. gelu 310p 2026-08-12 (29/29 T1, det 29/29). Cross-ref [PB-105](dav_m200_datacopypad_unsupported_both_directions.md)/[PB-108](dav_m200_duplicate_uint8_rejected.md)/[PB-113](dav_m200_shiftleft_shiftright_silent_noop.md), [OL-80](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md). [dav-m200/Ascend310P3, vector-compute, porting]

- Former A5Ops ID: `PB-112`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a (SDK impl defect); op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (gelu 310p, 2026-08-12 — on-device stage-dump bisect)`
`unverified_on: soc=Ascend310B, Ascend910, Ascend950PR (cmpmask SPR advance is a per-SoC impl detail — other SoCs' Select paths are separate code; OL-105's ShiftLeft/Select suspicions were 351x-side hypotheses, not this)`

- **Severity**: HIGH — compiles clean, runs without error, and silently produces wrong values for every element i ≥ 64 of a Compare/Select tile.
- **Symptom**: a `Compares`/`Select` pair (Level-2 count form) produces outputs as if the compare mask REPEATED every 64 elements — element i≥64 is selected by mask bit i%64.
- **Mechanism**: the dav-m200 Level-2 count impl (`VselImpl` in CANN 安装树内 impl/dav_m200/kernel_operator_vec_cmpsel_impl.h) does `set_cmpmask(sel)` ONCE, then a single `vsel(..., repeatTime=1, ...)` under `SetMaskCount(0, count)`. On device the cmpmask SPR does NOT advance per 64-element repeat, so the mask applies with period 64. Verified by stage-dump bisect (13 stages, one extra build cycle): a mask-reuse-period-64 simulation matched the observed outputs with match=1.0000.
- **Mitigation (preference order)**: (1) eliminate Compare/Select on dav-m200 entirely — mask-free arithmetic chain (clamp tail-kill + arithmetic sign blend; see CAND-M200-CANCELLATION-FREE-ARITHMETIC-SELECT) or Gather-based select (CAND-DAV-M200-GATHER-SELECT); (2) if a Select is unavoidable, keep count ≤ 64 per call.
- **Triage note**: do NOT declare "platform limitation" and stop at the first Select-based failure — the period-64 signature is provable on-device with a stage-dump bisect (dump intermediates per stage, simulate candidate mask semantics against the dumps) in one extra build cycle.
- **Evidence**: gelu 310p (2026-08-12, port_a3_to_a5 TARGET=310p) — failures_ledger precision iter 2: max_abs_diff_large from period-64 mask reuse; root-caused by the on-device stage-dump bisect (period-64 simulation match=1.0000), fixed by deleting Compare/Select entirely → 29/29 Tier-1 PASS, det 29/29 bit-identical. Kernel anchor 内部路径（不公开）workspace/gelu/kernel/gelu_kernel.h:149 (workspace scratch, gitignored).
- **Other instances (predicted)**: any dav-m200 kernel using `Compares`/`CompareScalar` + `Select` at tile counts > 64 — sign blends, clamp-via-select, predicate masks (VSEL_CMPMASK_SPR is the default cmpmask path). Cross-ref [PB-105](dav_m200_datacopypad_unsupported_both_directions.md)/[PB-108](dav_m200_duplicate_uint8_rejected.md)/[PB-113](dav_m200_shiftleft_shiftright_silent_noop.md) (the dav-m200 missing/broken-API class — grep the per-SoC impl header per OL-80 before using a new API), CAND-DAV-M200-GATHER-SELECT, CAND-M200-CANCELLATION-FREE-ARITHMETIC-SELECT.
