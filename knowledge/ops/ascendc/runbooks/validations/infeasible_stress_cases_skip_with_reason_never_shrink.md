---
type: Runbook
title: "Physically infeasible generated stress cases go through skipped_cases with an explicit hardware-limit reason — never hand-shrink the case set, never round the count up"
description: "Physically infeasible generated stress cases go through skipped_cases with an explicit hardware-limit reason — never hand-shrink the case set, never round the count up"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-406"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-406 — OL-406: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (harness / case_gen workflow); cann=all; bisheng=n/a; op_class=superlinear-output ops (diag-style N→N² expansion, outer products, pairwise kernels) — and any op whose input domain is unbounded while output is O(input²)。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (diag_v2 310p, 2026-08-09)。

**OL-406** — **physically infeasible generated stress cases go through `skipped_cases` with an explicit hardware-limit reason — never hand-shrink the case set, never round the count up.** case_gen explores input space, not output footprint; superlinear-output ops (N→N² diag, outer product, pairwise) can get a well-formed input with an infeasible output (n=1048576 diag → 1 TiB). Skip-with-reason preserves the coverage-gate audit; hand-shrinking is the P0abd fraud shape; "8/8 + 1 skip", not "9/9". Discriminator vs OL-168: that bans skipping FAILING cases; this scopes the one legitimate skip (never-runnable, reason-recorded). diag_v2 310p 2026-08-09. backend=all.

- Former A5Ops ID: `OL-406`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->

`applies_to: soc=all (harness / case_gen workflow); cann=all; bisheng=n/a; op_class=superlinear-output ops (diag-style N→N² expansion, outer products, pairwise kernels) — and any op whose input domain is unbounded while output is O(input²)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (diag_v2 310p, 2026-08-09)`

**Principle**: a signature/shape-driven case generator explores the INPUT space, not the
output's memory footprint. For ops whose output grows superlinearly in the input, the pilot
tier can emit a case that is a well-formed input but a physically infeasible output (1-D
n=1048576 diag → 2^40 B ≈ 1 TiB output tensor). The correct channel is the harness's
`skipped_cases` path with an EXPLICIT hardware-limit reason recorded per case — the coverage
gate counts skipped_cases, so honesty there preserves the audit trail. Two forbidden
shortcuts: (a) hand-shrinking the generated case set so the infeasible case never appears —
that silently reduces the coverage the gate thinks it counted (the P0abd coverage-fraud
shape, OL-168); (b) reporting the run as "N/N" when it is "(N−1)/(N−1) runnable + 1
hardware-limit skip" (P7 — the skip stays in the denominator). The discriminator vs the
OL-168 trap: that entry bans skipping a FAILING case to claim PASS; this entry scopes the
ONE legitimate use — a case that was never runnable on the hardware, recorded with its
reason.

**Concrete anchor**: diag_v2 310p `workspace/diag_v2/verification.json` (workspace scratch,
gitignored) — 8/8 runnable cases bit-exact T1 + case 2 (n=1048576, 1 TiB output) recorded
as a hardware-limit skip; reported as "8/8 + 1 skip", not "9/9".

**Evidence**: diag_v2 310p (2026-08-09, first instance): 9-case 1-D int8 k=0 pilot tier,
case 2 infeasible → skipped with explicit reason; 8/8 bit-exact T1, det 8/8.

**Other instances (predicted)**: outer-product / pairwise-distance / expansion ops on any
SoC; any port whose upstream input domain is unbounded but whose output is O(input²); also
input-side infeasibility (a generated case exceeding HBM before the kernel even runs).

**Cross-ref**: OL-168 (P0abd — the failing-case-skip fraud this entry is scoped against),
CAND-KERNEL-CONTRACT-NARROWING-INPUT-GEN (the sibling situation where input_gen must MODEL
a narrowed contract rather than skip it). backend=all.
