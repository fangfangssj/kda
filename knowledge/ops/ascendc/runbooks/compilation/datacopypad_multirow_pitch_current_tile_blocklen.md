---
type: Runbook
title: "multi-row `DataCopyPad` GM→UB — pitch staged rows by the CURRENT tile's `blockLen` (buffer sized by max is fine; indexing by the max corrupts a short final tile)"
description: "multi-row `DataCopyPad` GM→UB — pitch staged rows by the CURRENT tile's `blockLen` (buffer sized by max is fine; indexing by the max corrupts a short final tile)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["EC-160"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-160 — EC-160: multi-row `DataCopyPad` GM→UB — pitch staged rows by the CURRENT tile's `blockLen` (buffer sized by max is fine;

## 适用场景

- 适用平台与范围：soc=Ascend950PR；CANN 9.1.0.B060；op_class=all（任意多行 DataCopyPad、blockCount>1、暂存逐 tile 长度变化的 [rows, len] 块的 kernel）。
- 验证来源：soc=Ascend950PR_9579；CANN 9.1.0.B060（38_CBAMBlock K3，2026-08-13；三组独立 fixture：修复后 precision 50/50 PASS_T1、旧 pre-O2.5 边界 fixture 171/171、当前 O2.5 fixture 141/141）。

`applies_to: soc=Ascend950PR; cann=9.1.0.B060; bisheng=n/a; op_class=all (any multi-row DataCopyPad, blockCount>1, staging a [rows, len] block whose len varies per tile)`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0.B060 — 38_CBAMBlock K3, 2026-08-13. Three separate fixtures, not one widened claim: (a) precision 50/50 PASS_T1 after the fix; (b) the older pre-O2.5 unbound edge fixture, 171/171; (c) the current case-bound O2.5 fixture, 141/141 (a different artifact from (b), not a regression of it).`
`unverified_on: soc=Ascend910_V220 (A2/A3 — DataCopyPad GM→UB semantics likely transfer; not retested. EC-23's DataCopyPad crash was measured on Ascend950PR; both V220 and later V351 carry positive UB→GM evidence elsewhere in this file, so no direction ban is asserted here.)`

A single call pattern that produces **silently wrong trailing lanes** with no compile error and no runtime error.

**Pitching rows by the MAX tile length instead of the current one.**
Tiling computes one `ubRow_ = AlignUp(L_ * sizeof(T), 32) / sizeof(T)` from `L_`, the *maximum* tile length, sizes the UB buffer from it, and then also *indexes* staged rows as `xl[r * ubRow_]`. But `DataCopyPad` pitches each row from the `blockLen` it is **actually handed** — i.e. from the current tile's `len`. The two agree for every full tile and diverge on a short **final** tile whenever `AlignUp(len*sizeof(T),32) != AlignUp(L_*sizeof(T),32)`.

- **Correction**: size the buffer with the max (that part is right and necessary); *index* with the tile's actual `len`.
  ```cpp
  // WRONG: pitch from the max tile length used to size the buffer
  uint32_t ubRow = AlignUp(L_ * sizeof(T), 32) / sizeof(T);
  // RIGHT: pitch from the length this call is handed
  uint32_t ubRow = AlignUp(len * sizeof(T), 32) / sizeof(T);
  ```
- **Fault shape**: the defect fires only on a **short final tile** whose current-tile aligned pitch differs from the max-tile aligned pitch (`AlignUp(len*sizeof(T),32) != AlignUp(L_*sizeof(T),32)`); full tiles are always clean. Witnesses: `H·W = 117` (13 × 9, no partial tile) clean; `H·W = 133` wrong only in its trailing lanes. These two observations are witnesses of that shape, not a general divisibility rule.

**Provenance**: promoted from LO candidate `CAND-DATACOPYPAD-PITCH-CURRENT-LEN`; former payload intended `EC-89`, reassigned to `EC-160` by current owner-block authority (89 sits in no registered block; 160 is the first free ID in the the executor EC block). One item, one ID — do not read the two numbers as two entries. Source: branch `origin/tilelang/level4-38-cbamblock-20260813` @ `50e3d0d86` (payload `output/npukernelbench/src/kernels/38_CBAMBlock/kb_merge_payload.md`; worker knowledge `output/npukernelbench/src/kernels/38_CBAMBlock/kb_handoff/payload/knowledge_update.md`, same branch — retained evidence paths, immutable at that commit).

**Cross-reference**: P-P98 (the non-aligned `DataCopyPad` pattern this entry's pitch rule extends), OL-167 (no host-side pad+narrow — the rule that forces the in-kernel path where this bites), CAND-PP83 (single-row `blockCount=1` last-tile form; the pitch error cannot arise there), EC-172 (the padding byte-limit sibling), EC-23 (the DataCopyPad crash, measured on Ascend950PR — direction/arch scope note; V220 and later V351 carry positive UB→GM evidence, so no direction ban).
