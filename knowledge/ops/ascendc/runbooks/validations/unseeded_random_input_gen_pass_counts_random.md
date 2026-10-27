---
type: Runbook
title: "Unseeded random input_gen makes per-run pass counts a random variable — diff per-case on seeded cases only; degenerate out-of-range draws are a skipped-case class, never passes"
description: "Unseeded random input_gen makes per-run pass counts a random variable — diff per-case on seeded cases only; degenerate out-of-range draws are a skipped-case class, never passes"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-320"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-320 — OL-320: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (harness/data-contract, not silicon); cann=all; bisheng=n/a; op_class=any op whose input_gen.py draws unseeded random tensors。
- 验证来源：2_FFN (2026-07-25/26, two full canonical runs: same kernel scored 16 vs 27 FAILs from draw variance alone)。

**OL-320** — unseeded random input_gen makes per-run pass counts a random variable (same kernel 16 vs 27 FAILs across runs); diff per-case on seeded cases only; degenerate out-of-range draws are a skipped-case class with dissection evidence, never counted as passes.

- Former A5Ops ID: `OL-320`
- Fusion disposition: `additive_new`

`applies_to: soc=all (harness/data-contract, not silicon); cann=all; bisheng=n/a; op_class=any op whose input_gen.py draws unseeded random tensors`
`verified_on: 2_FFN (2026-07-25/26, two full canonical runs: same kernel scored 16 vs 27 FAILs from draw variance alone)`

**Principle**: if `input_gen.py` (or `get_input_groups`) draws `torch.rand`/`randn` without a fixed seed, then the verify pipeline, the canonical O5 eval, and the determinism check each run on DIFFERENT case data. A pass count is then a property of the draw, not of the kernel: the same kernel scored 16 FAILs in one canonical run and 27 in the next, because degenerate draws (mu +/- 100, sigma 25 -> fp16 overflow) hit different cases each run. Within one process the triple (cpu / npu / cand) is still consistent — it is cross-process comparisons that silently stop being comparisons of the same thing.

**Concrete anchor**: 2_FFN `get_input_groups` uses unseeded `torch.rand` branches plus random mu/sigma per tensor. Two seeded-invariant cases (6/7) measured byte-identical metrics across both runs (1.17e-2 / 4.46e-6), proving part of the matrix is seeded and the rest is not.

**Rules**: (1) before comparing two pass counts, verify the input generator is seeded — if not, only per-case diffs on seeded cases are meaningful; (2) a flaky-looking pass-count regression with no kernel change is a seed question first, a kernel question second; (3) degenerate draws that push ~97-100% of outputs out of dtype range form a distinct skipped-case class (ungradeable against a reference that itself outputs 97% Inf / 100% NaN there) — classify them with live dissection evidence, do NOT count them as passes, and do NOT chase them as bugs.

**Evidence**: 2_FFN (2026-07-25/26): run1 16 FAILs vs run2 27 FAILs, same kernel, draw variance only; degenerate-draw cases 6/10/20 classified skipped with ref-NaN-count dissection (ours saturates to Inf per fp16 contract, finite-sliver MERE 2.5e-4 where gradeable).

**Other instances (predicted)**: any benchmark-mode op with random input generation (most level3/level4 ops); determinism-check design (det must run both rounds inside ONE process or on seeded data); harness input_gen templates should emit a fixed seed by default (harness-level follow-up).
