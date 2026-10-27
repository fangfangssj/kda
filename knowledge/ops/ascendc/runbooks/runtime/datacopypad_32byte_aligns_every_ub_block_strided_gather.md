---
type: Runbook
title: "DataCopyPad 32-byte-aligns every block it writes into UB, so a strided multi-row gather is contiguous only when blockLen is a multiple of 32 bytes"
description: "DataCopyPad 32-byte-aligns every block it writes into UB, so a strided multi-row gather is contiguous only when blockLen is a multiple of 32 bytes"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-513"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-513 — OL-513: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=any AIV kernel that gathers strided rows into UB and then computes on them; build=any。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 — 4-42 CoTAttention im2col fusion, main 2026-09-05 (an intermediate build carrying this gather differed on 47/50 cases; that build also carried a separate addressing defect, so the count is not attributable to the gather alone, and the per-case split against row-byte alignment was not recorded)。

**OL-513** — **`DataCopyPad` 32-BYTE-ALIGNS EVERY BLOCK it writes into UB, so a strided multi-row gather is contiguous ONLY when `blockLen` is a multiple of 32 bytes** — for any other row width the rows land individually padded while the vector ops that follow read the buffer as contiguous, which is wrong for some shapes and silently right for others.

- Former A5Ops ID: `OL-513`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=any AIV kernel that gathers strided rows into UB and then computes on them; build=any`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 — 4-42 CoTAttention im2col fusion, main 2026-09-05 (an intermediate build carrying this gather differed on 47/50 cases; that build also carried a separate addressing defect, so the count is not attributable to the gather alone, and the per-case split against row-byte alignment was not recorded)`
`unverified_on: other SoCs and CANN versions — the 32 B block granularity is the documented UB access unit on this part, but it was measured here only; re-check before assuming the same constant elsewhere`

- **Category**: data movement / silent correctness
- **Trigger**: you are about to write `DataCopyPad(ubTensor, gmTensor[...], {rows, blockLen, srcStride, 0, 0}, pad)` with `rows > 1` and then run vector ops over `rows * elemsPerRow` elements as if they were contiguous.
- **Principle**: `blockCount > 1` pads **each block** up to a 32 B boundary in UB. The rows therefore sit `AlignUp(blockLen, 32)` apart, not `blockLen` apart. Any subsequent `Add`/`Mul`/`Cast`/reduction over the flat extent reads the inter-row padding as data.
- **Why it is hard to see**: it fails **only** for row widths where `blockLen % 32 != 0`. On a benchmark with mixed shapes that presents as "some cases pass, some fail", which reads as a numerical problem in whatever you just changed — not as an unsound access pattern. In 4-42 an intermediate fused build carrying this gather differed on 47 of 50 cases — but that same build still carried a separate addressing defect, and an earlier build with only that other defect differed on 47/50 too, so the count does not isolate this cause. Which specific widths matched was not recorded; the alignment rule below predicts the split, it was not measured case-by-case.
- **The safe idiom (and why the existing one works)**: use the **same blocking for the load and the store** — `load {rows, len, srcGap, 0, 0}` then `store {rows, len, 0, dstGap, 0}` — so the UB padding cancels exactly. That is why a GM→UB→GM strided copy (im2col's fast path) is correct at any width. It is *only* when the gathered data is **consumed** by vector math that the padding becomes visible.
- **Remedies when the data must be consumed**: (a) compact with a separate round-tripping copy kernel and let the consumer read contiguous memory; (b) keep the padded extent and size every other operand the same way so the layouts agree; (c) gather one row per call at 32 B-aligned UB offsets — correct but pays a call per row.
- **Detection**: compare **bit-exactly** against the path being replaced rather than relying on the precision profile. A tolerance gate blurs a shape-dependent defect into a scatter of failures; a bitwise control turns it into "these differ, those do not", and the split is the diagnosis.
- **Evidence**: main 2026-09-05, Ascend950DT_9582 / CANN 9.1.0, 4-42 CoTAttention. **Not a single-variable result — do not read the numbers below as this defect's effect size.** The fused path was brought up through four distinct defects, recorded in `TERMINAL_42_PAIRED.md` ADDENDUM 42: (1) a base-offset fault (device 507015), (2) a pitch parameter reused for other operands — 47/50 differing, (3) **this** gather, on the build after (2) was fixed — still 47/50 differing, (4) a K-tile divisor — 1/50 differing. Replacing the gather with a round-tripping compaction kernel is what moved the run off 47/50, and that is the observation supporting this entry. The terminal **50/50 bit-identical** / `VERDICT=BIT_NEUTRAL` result and the full-50 device-event GM **1.034612** (precision 50/50, DSO `08e1942cdb62b807c736d3db206c8dd861aa9d33e57d232da44083b8bf3fde61`) belong to the **final build containing the compaction and every other fix**, and are cited here only to show the corrected path is sound — no part of that gain is attributed to this entry.
- **Other instances (predicted)**: any kernel that adopts a strided gather to read a padded or blocked intermediate — conv-style layouts, transposed reads, any "read the interior out of a padded buffer" step. Not verified elsewhere.
- **Cross-ref**: OL-508 (same lane, cube path), P-P47 (the shared Load/Store run idiom). **Adjacent
  `DataCopyPad`/alignment entries, listed without adjudication** — this entry is GM→UB and silent-
  layout, so it neither supports nor refutes any of them: EC-23 (UB→GM 507035; its title and its
  three later cross-op bullets disagree, and its `Ascend950PR_9589` evidence line is not reconciled
  with P-P98's "V220-only" scope note), OL-167 (`DataCopy` **count** form silently truncates; forbids
  host-side pad+narrow), P-P98 (`DataCopyPad` + `DataCopyExtParams` as the sanctioned non-aligned
  tail write). **The EC-23 ↔ P-P98 boundary is UNRESOLVED on main; do not read this list as
  resolving it.** backend=ascendc.
