---
type: Runbook
title: "arch35 AIV GetBlockIdx() already folds in the sub-block — derive lane and row-half from one flat GetBlockIdx with blockDim = halves*lanes"
description: "arch35 AIV GetBlockIdx() already folds in the sub-block — derive lane and row-half from one flat GetBlockIdx with blockDim = halves*lanes"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-294"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-294 — OL-294: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR (all revisions); arch=3510; cann=9.1.T500; op_class=any vector kernel that splits work by (lane, row-half)。
- 验证来源：Ascend950PR_958b / CANN 9.1.T500 — purpose-built probe, KERNEL_TYPE_AIV_ONLY at blockDim 2 and 4。

**OL-294 — arch35 AIV `GetBlockIdx()` ALREADY folds in the sub-block** (`get_block_idx()*get_subblockdim()+get_subblockid()`), so lane-from-GetBlockIdx + row-half-from-GetSubBlockIdx double-uses it. **MEASURED on Ascend950PR_958b / CANN 9.1.T500 under AIV_ONLY: `GetSubBlockIdx()` is ALWAYS 0 and `GetTaskRation()` is 1**, so nothing doubles and instead every lane is pinned to half 0 — the upper half of every tile is never written (single-chunk finite-but-wrong, multi-chunk NaN; a 56-case suite went 1->17 on the fix and back to 1 on revert). Fix: derive lane AND half from one flat GetBlockIdx with `blockDim = halves*lanes`. **MIX row of that probe is INCONCLUSIVE** (empty AIC half, contradicts a real-MIX trace) — do not extrapolate. [arch35, AIV, AIV_ONLY, MIX_AIC_1_2, work-split, GetSubBlockIdx, GetTaskRation, linear-attention, NaN, stale-workspace]

- Former A5Ops ID: `OL-294`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR (all revisions); arch=3510; cann=9.1.T500; op_class=any vector kernel that splits work by (lane, row-half)`
`verified_on: Ascend950PR_958b / CANN 9.1.T500 — purpose-built probe, KERNEL_TYPE_AIV_ONLY at blockDim 2 and 4`
`source: asc/impl/basic_api/dav_3510/kernel_operator_sys_var_impl.h — GetBlockIdxImpl / GetSubBlockIdxImpl / GetTaskRationImpl`
**Mechanism.** On arch35 the AIV branch of `GetBlockIdx()` returns
`get_block_idx() * get_subblockdim() + get_subblockid()` — the sub-block is ALREADY
folded into the value. The AIC branch returns `get_block_idx()` plain. So taking the
lane from `GetBlockIdx()` and the row-half from `GetSubBlockIdx()` uses the sub-block
twice.
**MEASURED — and this is the half that reading the formula does not give you.**
Under `KERNEL_TYPE_AIV_ONLY`, blockDim 2 and 4 alike:
| form | `GetBlockIdx()` | `GetSubBlockIdx()` | `GetTaskRation()` | `GetBlockNum()` |
|---|---|---|---|---|
| AIV_ONLY, blockDim=2 | 0,1 | **0,0** | **1** | 2 |
| AIV_ONLY, blockDim=4 | 0,1,2,3 | **0,0,0,0** | **1** | 4 |
`get_subblockdim()` is **1**, so the formula collapses to `blockIdx` and **nothing
doubles**. The bug therefore takes a different shape than the formula implies:
`GetSubBlockIdx()` is constantly 0, so a row-half selector built on it pins EVERY
lane to half 0 and **the upper half of every tile is never written at all**.
**Counter-example — a guard must cover BOTH shapes, they present differently:**
- **AIV_ONLY** (measured): `sub` always 0 → only rows `0..N/2` computed; the rest of
  each tile keeps whatever the workspace held.
- **MIX_AIC_1_2** (NOT settled — see scope): were `get_subblockdim()` 2, the index
  WOULD double and an `if (idx >= NBH) return;` guard would also make half the blocks
  exit immediately. Same root confusion, unrecognisably different symptom.
**Observed consequence, and why it misleads.** A chunked linear-attention forward
carrying this bug gave single-chunk cases FINITE BUT WRONG (stale workspace in the
unwritten half) and multi-chunk cases **NaN** (the recurrence consuming a
never-written state half). Fixing it moved a 56-case suite 1 → 17 passes; reverting
this one expression returned it to exactly 1. Load-bearing, not cosmetic.
**Rule.** Derive BOTH coordinates from one flat index — `lane = GetBlockIdx() / H`,
`half = GetBlockIdx() % H`, host launching `blockDim = H * lanes`. That depends on
neither `GetSubBlockIdx()` nor `get_subblockdim()`, so it is correct under either
task type.
**SCOPE, deliberately narrow.** The table is `KERNEL_TYPE_AIV_ONLY` only. The same
probe's MIX row is **INCONCLUSIVE**: its body sat entirely inside `if ASCEND_IS_AIV`
so the AIC half was empty and the TU may have been configured as effectively
AIV-only — and that row CONTRADICTS an earlier trace from a real MIX TU in which two
sub-blocks wrote distinct markers, i.e. `sub=1` genuinely existed. Do not extend this
card to MIX until a probe with a non-empty AIC half settles it.
**Method note, worth more than the fact.** The first diagnosis here was "the
sub-block is counted twice", read off the source formula WITHOUT measuring
`get_subblockdim()`. It predicted both observed symptoms correctly and was still
wrong — two different mechanisms converge on the same consequence (an unwritten
half). **A correct prediction does not validate the reasoning that produced it: they
are separate claims, and only one of them had been tested.** Read the formula, then
measure the quantities it depends on.
**Cross-ref**: `fa_class/cross_core_sync.md` §7 (arch35 core/pipe semantics),
CAND-SCAN-FP32-ACCUM (accumulation rule for this op class), CAND-A5OPS-0011 (neighbouring
operational entry).
