---
type: Runbook
title: "Upstream tiling-key variant sprawl is perf-only, not correctness — for a mathematically-uniform op a single general-case kernel suffices for the correctness-verify path"
description: "Upstream tiling-key variant sprawl is perf-only, not correctness — for a mathematically-uniform op a single general-case kernel suffices for the correctness-verify path"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-410 — OL-410: A5Ops additive extension

> 关联：同一 OL-410 主题已由既有卡 [ol_410_upstream_tiling_key_variant_sprawl_is_perf_only.md](../performance/ol_410_upstream_tiling_key_variant_sprawl_is_perf_only.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=data-movement / scatter-gather / index (generalizes to any mathematically-uniform op)。
- 验证来源：soc=Ascend310P3 (dav-m200); cann=9.0.0。

**OL-410** — **upstream tiling-key variant sprawl is PERF-only, not correctness — for a mathematically-uniform op (pure copy/assignment, no per-variant algorithm divergence) a single general-case kernel suffices for the correctness-verify path; author from A3 algorithm source, don't clone the variant tree.** Bit-exact-by-construction sub-principle: a copy-only/no-Cast/no-FP data-movement kernel is bit-exact by construction — a non-zero `max_abs_diff` is always an indexing/copy bug (cf. OL-167/OL-407), never a precision ceiling to tolerance-loosen (Cast-bearing data movement → OL-44). Scope: does NOT apply when tiling keys encode genuinely different algorithms (reduction order, transpose layout). Anchor `workspace/scatter_list/kernel/scatter_list_kernel.h` (15 variants → 1 ~150-line kernel). Evidence: scatter_list 310p 2026-08-09, 12/12 bit-exact. Cross-ref [OL-37](../../optimizations/ol_37_data_movement_vecin_vecout_bridge.md), [OL-44](../compilation/ol_44_cast_bf16_f32_bf16_roundtrip_is_not_necessarily_bi.md), [OL-167](../compilation/ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md), [OL-407](../compilation/ol_407_datacopy_count_is_an_element_count_on_every_arch.md), [OL-160](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md). backend=ascendc.

- Former A5Ops ID: `OL-410`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=data-movement / scatter-gather / index (generalizes to any mathematically-uniform op)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5), Ascend910_220x (A3) — this is an algorithmic-design rule (arch-independent), but evidenced only on dav-m200; the principle transfers, re-confirm the "general kernel suffices" claim per-op on other SoCs.`

**Principle**: upstream CANN op sources frequently carry a large family of tiling-key variants (e.g. `TransposeSmall/Large`, `NegLarge/LargeE`, `RSBSE/RLBSE`, dim-2-large, …) — often dozens of variants and hundreds of KB of code. These exist to pick a fast code path for a specific shape regime; they encode PERFORMANCE specializations, not the operation's mathematical definition. When the operation is mathematically uniform (the same transform applies regardless of shape — e.g. `reduce="update"` is pure assignment, a copy), the correctness-verify path does NOT need to replicate the variant tree: a single general-case kernel that distributes work across cores and moves/computes the data correctly produces bit-identical output to any of the variants, because the variants differ only in HOW they move/compute the same bytes, not in WHAT result they produce. Author the generated kernel from the A3 *algorithm source* (the math), not by cloning the variant tree.

**Why correctness is preserved — bit-exact-by-construction (folded sub-principle)**: a pure data-movement kernel — copy-only, no `Cast`, no floating-point op — is bit-exact with the reference BY CONSTRUCTION: it relocates bytes without transforming a bit. So for such ops a non-zero `max_abs_diff` is NEVER a "precision ceiling" to tolerance-loosen — it is always a real bug (wrong index, OOB, a missed copy, a truncated DataCopy per OL-167/OL-407). Diagnostic flip-side: when your data-movement kernel is not bit-exact, chase the indexing/copy bug, do NOT reach for a tolerance. (For data movement WITH a dtype `Cast`, see OL-44 — the cast roundtrip is the one place a data path is not automatically bit-exact.)

**Scope boundary — do NOT over-generalize (regression-risk)**: this applies ONLY when the operation is mathematically uniform across the tiling keys. It does NOT license collapsing variants when different tiling keys encode GENUINELY DIFFERENT algorithms (different reduction strategies, transpose-vs-straight layouts, approximate-vs-exact paths). Test the claim: if removing the variant tree changes output on ANY shape, the variants were correctness-relevant and this rule does not apply. The verify-path simplification is also orthogonal to the "generate from A3 source, do not wrap arch35" mandate — the simplified kernel is still GENERATED from the algorithm, not copied from upstream.

**Concrete anchor** (scatter_list 310p, `workspace/scatter_list/kernel/scatter_list_kernel.h`): the 220x `scatter_list_base.h` source carries ~15 tiling-key variants (~200 KB). The dav-m200 generated kernel replaces ALL of them with one ~150-line class: distribute (b,n) pairs across the cores, copy var→UB→varOut, conditionally copy updates→UB→varOut. Correct for `reduce="update"` because assignment is shape-invariant. 12/12 cases `max_abs_diff=0.0`.

**Evidence**: scatter_list 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, ACLRT_LAUNCH_KERNEL+pybind verify path) — 12/12 bit-exact. `[first evidence]`

**Other instances (predicted)**: scatter_add (when the reduce semantics are uniform), index_copy/index_put pure-assignment paths, gather/gather_elements copy-only paths — any "many upstream tiling variants but the math is a copy/assignment" op. **Counter-examples (rule does NOT apply)**: reduction-class ops whose tiling keys choose different accumulation orders; transpose-class ops where layout-affecting variants change the result's memory order.

**Cross-ref**: OL-37 (data-movement VECIN→VECOUT bridge — the in-kernel pipe pattern such a kernel still needs), OL-44 (cast roundtrip — the non-bit-exact data-movement case), OL-167/OL-407 (DataCopy count semantics the simplified kernel must still get right), OL-160 (canonical verify-path entry-point contract).
