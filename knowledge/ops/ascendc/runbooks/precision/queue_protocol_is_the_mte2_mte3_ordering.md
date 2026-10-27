---
type: Runbook
title: "On V351 AIV the TQue/TQueBind protocol IS the MTE2→MTE3 ordering mechanism — neither removing the VECOUT re-enqueue nor hand-rolled flag ping-pong works"
description: "On V351 AIV the TQue/TQueBind protocol IS the MTE2→MTE3 ordering mechanism — neither removing the VECOUT re-enqueue nor hand-rolled flag ping-pong works"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-QUEUE-PROTOCOL-IS-THE-MTE2-MTE3-ORDERING"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-QUEUE-PROTOCOL-IS-THE-MTE2-MTE3-ORDERING — On V351 AIV the TQue/TQueBind protocol IS the MTE2→MTE3 ordering mechanism — neither removing the VECOUT re-enqueue nor hand-rolled flag ping-pong works

## 适用场景

- 适用平台与范围：soc=Ascend950PR (351x family)；CANN 9.1.0；经 TQue/TQueBind 暂存的纯拷贝/数据搬运 kernel。
- 验证来源：soc=Ascend950DT_9582、CANN 9.1.0（13_Cat，2026-09-16）；soc=Ascend910_V220 (A3) 与 dav-m200 不适用——该 SoC 家族需要相反构造（见正文），不可跨平台套用。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**CAND-QUEUE-PROTOCOL-IS-THE-MTE2-MTE3-ORDERING** — for a DataCopyPad copy engine staged through `TQueBind<VECIN,VECOUT>` on V351 AIV, cross-pipe MTE2→MTE3 ordering is carried by the queue protocol itself: `DeQue<VECIN>` chains MTE2→VEC, and the re-enqueue as VECOUT is what transitively gates the MTE3 leg (and tracks the slot's MTE3 use for reuse). Two measured anti-patterns, both producing garbage that looks like a kernel bug:
1. Removing the VECOUT re-enqueue as "overhead reduction" (bare `TQue<VECIN,N>` passthrough, DeQue → DataCopyPad-out directly) loses the cross-pipe ordering → garbage on mid/large cases.
2. Replacing the queue idiom with manual `SetFlag/WaitFlag<HardEvent::MTE2_MTE3>` / `<MTE3_MTE2>` ping-pong does NOT reproduce the ordering on V351 AIV — bit-mismatch even at 1 tile/core with blockCount>1.
Do not hand-roll flag-based MTE2↔MTE3 pipelines for DataCopyPad copy engines on this platform; use the TQue/TQueBind protocol and treat its enqueue pairs as semantics, not overhead.

`applies_to: soc=Ascend950PR (351x family); cann=9.1.0; bisheng=n/a; op_class=pure-copy / data-movement kernels staged through TQue/TQueBind`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 (13_Cat, 2026-09-16)`
`unverified_on: soc=Ascend910_V220 (A3); soc=Ascend310P3 / dav-m200 — DIVERGENT: on dav-m200 manual MTE2_MTE3 flags are REQUIRED, not decorative (CAND-DAV-M200-VECOUT-MTE2-MTE3-FLAGS). The two SoC families need OPPOSITE constructions; never cross-apply.`

- **Concrete anchor**: `TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> queue;` — the `EnQue`/`DeQue` pair across the VECIN→VECOUT bind is the sync; deleting the VECOUT half for a "cheaper" passthrough is the exact failing diff.
- **Evidence (anti-pattern 1, REVERT)**: 13_Cat ko-2 Opt1 (2026-09-16): `TQueBind<VECIN,VECOUT,2>` → bare `TQue<VECIN,4>` passthrough → precision FAIL, garbage on mid/large cases (MERE=inf). Reverted.
- **Evidence (anti-pattern 2, direction abandoned)**: 13_Cat ko-2 probe A: manual 2-slot flag ping-pong → bit-mismatch even at 1 tile/core, blockCount>1.
- **Side effect — closes an open verification**: OL-49 / P-P36 record TQueBind as "UNVERIFIED on A5". This op's shipped kernel (51/51 OVERALL_T1 bit-exact + det 51/51 on Ascend950DT_9582 / CANN 9.1.0) is that verification instance; an in-place status flip of OL-49/P-P36 is the follow-up edit.
- **Other instances (predicted)**: any pure-copy or copy-dominated kernel (cat/split/pad/transpose-copy stages) considering queue-structure "simplification" or flag-based pipeline rewrites on V351.
- **Cross-ref**: OL-49, P-P36 (TQueBind idiom), OL-94 (TQue implicit-sync decision table), CAND-DAV-M200-VECOUT-MTE2-MTE3-FLAGS (opposite-SoC construction — do not generalize across). backend=ascendc.

**UPDATED 2026-09-17 — CONTRADICTION EVIDENCE (13_Cat M-I run-1 ko-2 L3, measured)**: a
DIFFERENT construction works bit-exact on the same platform — `TBuf` 2-slot double buffer +
per-slot (fwd MTE2_MTE3, bwd MTE3_MTE2) event pairs with literal ids 0-3, GM->UB->GM pure copy,
NO V-pipe pass, 51/51 bit-exact all dtypes. This narrows (not reverses) the claim above: the
measured failure was the specific 2-slot flag ping-pong replacing the TQueBind idiom in a
DataCopyPad copy engine; the TBuf per-slot bidirectional event-pair construction is proven
working. The safe guidance becomes: prefer the TQue/TQueBind protocol; if hand-rolling, the
TBuf + per-slot event-pair shape is the measured-working one; the bare ping-pong replacement
of TQueBind remains measured-broken.
