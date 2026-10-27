---
type: Runbook
title: "An instrument that changes memory layout is not the build under study — state which invariants it preserves at the declared granularity and which it does not"
description: "An instrument that changes memory layout is not the build under study — state which invariants it preserves at the declared granularity and which it does not"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-298"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-298 — OL-298: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; op_class=all。
- 验证来源：host-only mechanised layout audit (g++, no CANN, no device), FlashKDA fwd workspace, 40 slabs x 5 shapes, 2026-07-31。

**OL-298** — **An instrument that changes memory layout is not the build under study: state which invariants it preserves AT THE DECLARED GRANULARITY and which it does not.** 512 B guards between 40 workspace slabs preserve the declared 512 B requirement (host-only checker derives the table twice, with guard and with guard = 0, and flags any slab not 512 B aligned in both — **0 violations, 40 slabs x 5 shapes**, growth +20480 B) but move **35 of 40** slabs' **mod-4096 residue** (starts shift 0..19968 B), so any undocumented 1 KiB / 4 KiB dependence is perturbed. The audit's own load-bearing control: 41/41 offsets derived from its size table agree with the C++ layout authority — without it every guard range is fiction. Keep the revert to one symbol.

- Former A5Ops ID: `OL-298`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; op_class=all`
`verified_on: host-only mechanised layout audit (g++, no CANN, no device), FlashKDA fwd workspace, 40 slabs x 5 shapes, 2026-07-31`
**Principle**: inserting guards / padding / canaries between allocations produces a
**different build**, so a result from it is a result about that build. "Alignment is
unchanged" is only true at the granularity someone wrote down; every stronger alignment
the code never declared is silently perturbed. Publish the preservation claim *with its
granularity*, and publish what moved.
**Measured.** The guard is 512 B — a multiple of the declared `FKDA_ALIGN` (512 B) — and
each slab start is aligned up **before** the guard is added, so every start shifts by a
multiple of 512.
- **Preserved at the declared granularity**: a host-only checker derives the slab table
  **twice** — once with the guard and once with guard = 0, which reproduces the original
  layout byte-for-byte — and flags any slab that is not 512 B aligned in **both**.
  Result: **0 violations across 40 slabs and 5 shapes** (`NT,NBH` = (4096,16), (8,2),
  (256,16), (1,1), (0,0)). Growth is +20480 B (40 x 512), identical at every shape.
- **NOT preserved**: **35 of 40** slabs' **mod-4096 residue moves**; starts shift by
  0..19968 B. Alignments above 512 B are not preserved and cannot be by any non-zero
  guard. Nothing in the layout header, the tiling or the kernels declares more than
  512 B — but an **undocumented** 1 KiB / 4 KiB dependence would be perturbed, and a
  result from the instrumented build could then differ for a reason unrelated to the
  defect being hunted.
**The control that makes the audit mean anything**: the checker also validates its own
size table against the C++ authority — **41/41** offsets (40 slabs + total) derived from
the table agree with the real layout function at every shape. Without that gate every
guard range it prints is fiction and the alignment claim describes an imaginary layout.
**Rule.** When instrumentation changes layout, publish (a) the declared granularity,
quoted from where it is declared; (b) a mechanised both-ways check that the declared
invariant holds in the instrumented AND the uninstrumented derivation; (c) an explicit
list of what did move. Keep the revert down to one symbol (here `FKDA_GUARD_BYTES = 0`)
so the instrumented build and the studied build are one edit apart.
**Cross-ref**: OL-297 (the same instrument's discriminating-power failure), OL-296
(agree with the authority on a known point before decomposing anything).
---