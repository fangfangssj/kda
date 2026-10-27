---
type: Runbook
title: "A grouped-GEMM group list is either cumulative or segmented; the default is cumulative and the wrong choice is silent (correct group 0, wrong everything after)"
description: "A grouped-GEMM group list is either cumulative or segmented; the default is cumulative and the wrong choice is silent (correct group 0, wrong everything after)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-315"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-315 — OL-315: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=grouped-matmul, MoE expert FFN。
- 验证来源：catlass source read on the A5 PR box; the MoonEP consequence is REASONED, not yet measured。

**OL-315** — **A grouped-GEMM "group list" is either CUMULATIVE or SEGMENTED; the DEFAULT is cumulative and the wrong choice is SILENT.** catlass `grouped_matmul_slice_m.hpp:141-148` gates it on `CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED`: defined ⇒ `list[i]` is group i's rows; **undefined (default)** ⇒ `list[i]-list[i-1]`. The two agree only on group 0, so a wrong convention gives a **correct group 0 and wrong everything after**, with no error and a plausible output. ⚠️ Bites MoE/MoonEP because the vendor dispatch exposes both shapes under similar names (`expert_token_nums`, `cu_seqlens`) and `npu_moe_distribute_dispatch` takes an `expert_token_nums_type` flag selecting between them ⇒ the same tensor is either convention depending on a flag layers away; **do not infer it from the variable name — read the flag or the consumer**. **Detection**: deliberately UNEQUAL group lengths + verify each group against ITS OWN operand; equal-size groups or an aggregate-only check cannot see it, since group 0 matches either way.

- Former A5Ops ID: `OL-315`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=grouped-matmul, MoE expert FFN`
`verified_on: catlass source read on the A5 PR box; the MoonEP consequence is REASONED, not yet measured`
**Principle.** Every grouped-matmul API takes a per-group length vector, and there are **two
incompatible conventions** for it: **segmented** (`list[i]` = rows of group i) and **cumulative**
(`list[i]` = running total, so group i is `list[i] - list[i-1]`). They coincide only for the
first group, so a wrong choice produces a **correct group 0 and wrong everything after** — with
no error, no shape mismatch, and a plausible-looking output.
**Measured in catlass** (`include/catlass/gemm/kernel/grouped_matmul_slice_m.hpp:141-148`):
```cpp
#ifdef CATLASS_EXPERIMENTAL_GROUPLIST_SEGMENTED
    uint32_t currentM = groupList.GetValue(groupIdx);                    // SEGMENTED
#else
    uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)   // CUMULATIVE  <-- DEFAULT
        : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
#endif
```
⇒ **The default build is CUMULATIVE**, and the segmented reading is behind an
`EXPERIMENTAL`-named macro. A caller who assumes "the list holds per-expert token counts" — the
natural reading of a name like `tokens_per_expert` — is wrong by default.
**Why this bites MoE / MoonEP specifically.** The vendor dispatch already exposes *both* shapes
under similar names (`expert_token_nums` and `cu_seqlens`), and `npu_moe_distribute_dispatch`
takes an `expert_token_nums_type` flag that selects between them. So the same tensor can be
either convention depending on a flag several layers away. **Do not infer the convention from
the variable's name; read the flag or the consuming code.**
**Detection.** Give the groups **deliberately unequal lengths** and verify **each group against
its own operand** — never a single fused norm over the whole output. Under the wrong convention
group 0 still matches, so a fixture with equal group sizes, or one that only checks an aggregate,
cannot see the fault at all. (Equal sizes are additionally degenerate here: cumulative and
segmented lists are then trivially distinguishable but the *outputs* still differ, so the test
looks discriminating while being blind.)
