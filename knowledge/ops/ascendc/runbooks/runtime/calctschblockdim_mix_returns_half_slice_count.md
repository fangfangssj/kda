---
type: Runbook
title: "CalcTschBlockDim for MIX returns half the slice count — it encodes an AIV-parallel decomposition; calling it on an AIC-parallel one silently drops half the cube work"
description: "CalcTschBlockDim for MIX returns half the slice count — it encodes an AIV-parallel decomposition; calling it on an AIC-parallel one silently drops half the cube work"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-293"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-293 — OL-293: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR_958b; arch=arch35; backend=ascendc; task_type=MIX_AIC_1_2。
- 验证来源：soc=Ascend950PR_958b; cann=9.1.T500 (FlashKDA host A/B 2026-07-30)。

**OL-293** — **`CalcTschBlockDim` for MIX returns HALF the slice count: it encodes an AIV-parallel decomposition, so calling it on an AIC-parallel one silently drops half the cube work** — measured `Ascend950PR_958b` (aic=32/aiv=64): slices 2/4/8/16 -> cube-only 2/4/8/16 but MIX **1/2/4/8**. Not a clamp: with `taskRation 1:2` it assumes the 2 paired AIVs consume 2 DIFFERENT slices, i.e. `sliceNum` is VECTOR-side work and `blockDim` is an AIC count. If slices are AIC-parallel (one cube block per slice, AIVs splitting rows WITHIN it — FlashKDA's `(b,h)` lanes) adopting it launches half the blocks and half the slices are never computed — no error, output = stale workspace. Ask **AIV-parallel or AIC-parallel?** before calling; AIC-parallel -> size on `GetCoreNumAic()`. Scopes OL-210's "call it, don't copy-inline". Cross-ref [OL-210](../compilation/ol_210_for_a_mix_cross_core_sync_op_host_launch_is_the_ca.md), [OL-294](arch35_aiv_getblockidx_already_folds_subblock.md) (device-side twin: same `taskRation` folds into AIV `GetBlockIdx()`).

- Former A5Ops ID: `OL-293`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR_958b; arch=arch35; backend=ascendc; task_type=MIX_AIC_1_2`
`verified_on: soc=Ascend950PR_958b; cann=9.1.T500 (FlashKDA host A/B 2026-07-30)`
OL-210 says to **call** `PlatformAscendC::CalcTschBlockDim` rather than copy-inline its
arithmetic. That is still right, and this card is the qualifier: **calling it is only
correct if your decomposition matches the one it assumes.**
Signature (`platform_ascendc.h:127`):
`CalcTschBlockDim(sliceNum, aicCoreNum, aivCoreNum)`.
Measured on a card reporting `GetCoreNumAic()=32`, `GetCoreNumAiv()=64`:
| `sliceNum` | cube-only (`aivCoreNum=0`) | MIX (`aic=32, aiv=64`) |
|---|---|---|
| 2  | 2  | **1** |
| 4  | 4  | **2** |
| 8  | 8  | **4** |
| 16 | 16 | **8** |
The MIX form **halves** it. That is not a bug and not a clamp — with `taskRation 1:2`
each AIC block is paired with 2 AIV sub-blocks, and the function assumes those two AIVs
consume **two different slices**. So `blockDim` counts AIC blocks, and 1 AIC block
covers 2 slices. `sliceNum` is being interpreted as *vector*-side work.
**The failure mode**: if your slices are **AIC-parallel** — one cube block per slice,
with the paired AIVs splitting rows *within* that slice — then adopting this value
launches half the cube blocks you need and **half your slices are silently never
computed**. No error, no exception; the output for the missing slices is whatever the
workspace held. FlashKDA maps each `(b, h)` lane to one AIC block with its 2 AIVs
splitting the 16 rows of a tile, i.e. AIC-parallel, and correctly uses
`min(NBH, GetCoreNumAic())` instead.
**The question to ask before calling it**: *are my slices AIV-parallel or AIC-parallel?*
- **AIV-parallel** (slice ↔ AIV sub-block; e.g. token- or expert-sliced grouped GEMM,
  elementwise epilogues): call it, it is correct and it is the vendor's own arithmetic.
- **AIC-parallel** (slice ↔ cube block; a per-lane recurrence, a batched matmul where
  the batch index selects the cube block): do **not** call it. Size on
  `GetCoreNumAic()` and let the AIVs subdivide within the slice.
**Why this is easy to miss**: the two cases differ by exactly a factor of 2, so a
2-slice job launches 1 block and *looks* like it merely under-parallelised. The
`sliceNum`-vs-`blockDim` distinction is real and the name does not carry it — `blockDim`
in a MIX launch is an **AIC** count, while `sliceNum` here is a **vector**-work count.
**Cross-ref**: OL-210 (call it, don't copy-inline — this card scopes when to call it at
all), OL-294 (the device-side twin: `GetBlockIdx()` on the AIV also spans
`blockDim × subblockdim`, so host-side halving and device-side folding are the same
`taskRation` showing up on both sides of the launch).
---
