---
type: Runbook
title: "on dav-m200, multi-core scalar GM stores are silently dropped under concurrency, even with per-core exclusive 32B datablocks"
description: "on dav-m200, multi-core scalar GM stores are silently dropped under concurrency, even with per-core exclusive 32B datablocks"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-124 — PB-124: A5Ops additive extension

> 关联：同一 PB-124 主题已由既有卡 [pb_124_on_dav_m200_multi_core_scalar_gm_stores_are_sile.md](pb_124_on_dav_m200_multi_core_scalar_gm_stores_are_sile.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.0.0；op_class=all（blockDim>1 且多于一个核发出 scalar-pipe GM 存储的 kernel，如按核暂存的 SetValue 尾）。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（group_norm_silu 310p，2026-08-17 — DEVICE-RUN，benchmark-acceptance lane；常量输入探针 + 按核普查调试 kernel，3/3 复现）。

**PB-124** — **on dav-m200, multi-core scalar GM stores are silently DROPPED under concurrency, even with per-core exclusive 32B datablocks** — 4 blocks scalar-storing to exclusive datablocks, one block's stores dropped 3/3 trials; not a flush delay; scalar GM LOADS unaffected. No fence fixes it (unlike the same-core, fenceable PB-123): the only safe rule is zero multi-core scalar GM stores — single-writer-per-32B-block ownership, aligned DataCopy-only stores. Device-proven justification for P-P201's block-0-only scalar tail. Unaligned DataCopy dst works standalone but not under concurrency — keep all GM addresses 32B-aligned, loads included. group_norm_silu 310p 2026-08-17 (DEVICE-RUN 68/68, det 68/68). Cross-ref [PB-123](dav_m200_scalar_store_mte3_sector_drops_32b_block.md), P-P201（A5Ops 绑定外部引用；本语料未收录目标）, CAND-M200-CROSS-PIPE-32B-BLOCK-RACE, [PB-105](dav_m200_datacopypad_unsupported_both_directions.md). [dav-m200/Ascend310P3, multi-core, silent-data-loss]

- Former A5Ops ID: `PB-124`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (blockDim>1 kernels where MORE THAN ONE core issues scalar-pipe GM stores, e.g. SetValue tails staged per core)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (group_norm_silu 310p, 2026-08-17 — DEVICE-RUN, benchmark-acceptance lane; constant-input probes + block-census debug kernels, 3/3 repro)`
`unverified_on: soc=Ascend910 (220x), Ascend950PR (351x) — do not transfer without re-probe`

- **Severity**: HIGH — silent store loss + run-to-run non-determinism; launch and sync return clean, the dropped region keeps its pre-launch bytes.
- **Symptom**: with 4 blocks each scalar-storing (`GlobalTensor.SetValue`) to *exclusive* 32B datablocks (no sharing, no MTE3 involvement), one block's stores were dropped 3/3 trials; with shared datablocks, random subsets of stores were dropped per run. Stable after 0.5 s + full sync — NOT a flush delay. Scalar GM **loads** are unaffected (multi-core gamma/beta `GetValue` loads verified across all 68 cases).
- **Scope clarification (vs PB-123 — read both before choosing a fix)**: PB-123 is a SAME-core ordering hazard (scalar store sharing a 64B sector with an in-flight MTE3 copy) that a `SetFlag/WaitFlag<HardEvent::MTE3_S>` fence DOES fix. PB-124 is a CROSS-core concurrency hazard on the scalar store path itself — no fence fixes it; the only safe rule is **no scalar GM stores from multiple cores, period**. Single-writer scalar stores remain OK: this PB is the device-proven reason P-P201's "block-0-only scalar tail" convention exists (one core scalar-stores, the rest use aligned DataCopy), upgrading that convention to a necessity. It also explains the PB-106-adjacent "multi-core scalar" folklore (CAND-DAV-M200-SCALAR-READ-STALE-MULTICORE is the read-side sibling).
- **Fix (shipped recipe)**: zero scalar GM stores in multi-core kernels. Single-writer-per-32B-block ownership: every GM output block is written by exactly one aligned `DataCopy`; ragged row head/tail handled by aligned loads + UB lane-picking, never by element-wise `SetValue`. Corollary on alignment: an element-unaligned `DataCopy` dst works STANDALONE on dav-m200 (probe: dst elem-offset 4 fp32 = byte 16 landed 32/32 values) but must NOT be trusted under concurrency (v1's unaligned stores + concurrent scalar stores to the same 32B block lost data) — keep ALL DataCopy GM addresses 32B-aligned, loads included.
- **Detection rule for future ops**: multi-core m200 kernel with per-core scalar GM stores + flaky large-diff holes that move across reruns → this class. Bisect with constant-input probes (each core stores a distinct constant to its exclusive region, census what landed) before theorizing about fences.
- **Concrete anchor**: `workspace/group_norm_silu/kernel/group_norm_silu_kernel.h` (v2 shipped form — block-ownership partition, aligned DataCopy-only stores; workspace scratch, gitignored).
- **Evidence**: group_norm_silu 310p (2026-08-17, Ascend310P3/dav-m200, CANN 9.0.0, benchmark-acceptance lane, DEVICE-RUN): v1 (scalar GM store tails + element-unaligned DataCopy dst) failed det with large/garbage diffs on 41/68 cases; on-device bisect isolated the multi-core scalar-store drop (exclusive-datablock probe, 3/3). v2 rewrite (zero scalar GM stores, single-writer ownership) → 67/68; remaining case-63 nondeterminism was the separate staged-buffer fence class (CAND-DAV-M200-STAGED-BLOCK-FENCE-SET) → 68/68 PASS, det 68/68 bit-identical, max_abs_diff 3.9e-3 (fp16 ULP), 6/68 bit-exact (`workspace/group_norm_silu/failures_ledger.md` precision iters 1-2, `workspace/group_norm_silu/verification.json`).
- **Other instances (predicted)**: any multi-core dav-m200 kernel tempted to use `SetValue` for ragged tails, per-core variable-length outputs, scalar scatter/epilogues — scatter tails, string/categorical encoders, ragged-batch norm reductions, per-core count-prefix writes.
- **Cross-ref**: PB-123 (the same-core MTE3↔scalar-store ordering sibling — fenceable, different fix), P-P201 (block-0 scalar tail convention — this PB is its justification), CAND-M200-CROSS-PIPE-32B-BLOCK-RACE (cross-core S-vs-MTE3 tear on a SHARED 32B block — PB-124 drops stores even on EXCLUSIVE blocks), PB-105 (no DataCopyPad — why tails are tempting here), CAND-DAV-M200-SCALAR-READ-STALE-MULTICORE (read-side sibling). backend=ascendc.

- **Evidence (2nd op, independent probe)**: reduce_sum 310p (2026-08-17, CANN 9.1.0) — dedicated multi-core probe reproduced the drop at blockDim 4/8 (~50% of disjoint 32B-separated scalar stores lost, run-varying, silent rc==0) while blockDim ≤ 2 landed reliably in the same probe; multi-core scalar LOADS 100% reliable (store-direction-only, matching above). Additional data points: `SetFlag/WaitFlag<HardEvent::S_MTE3>` + trailing dummy MTE3 copy does NOT fix it, and `DataCacheCleanAndInvalid(ENTIRE_DATA_CACHE)` is an UNDEFINED SYMBOL on m200 (declared, not implemented — PB-105/PB-110 missing-API class). Shipped fix: all output via UB + vector DataCopy + host quantum-multiple over-alloc/narrow; 37/37 tier-1 PASS.
