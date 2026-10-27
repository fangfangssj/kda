---
type: Runbook
title: "A vendor reference op diverging from its own documented spec makes inline ours-vs-ref measure the instrument's bug — the two-tier ladder (T1 vs CPU truth, T2 ours<=ref against truth) is the only sound adjudication"
description: "A vendor reference op diverging from its own documented spec makes inline ours-vs-ref measure the instrument's bug — the two-tier ladder (T1 vs CPU truth, T2 ours<=ref against truth) is the only sound adjudication"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-319"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-319 — OL-319: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (oracle/grader design, not silicon); cann=all; bisheng=n/a; op_class=any op verified against a vendor reference op。
- 验证来源：2_FFN bf16 inner_precise=0 + fp16 ip=1 cases (2026-07-25/26, Ascend910_9382, CANN 9.0.0)。

**OL-319** — vendor reference op diverging from its own documented spec ⇒ inline ours-vs-ref measures the instrument's bug; the two-tier ladder (T1 vs CPU truth, T2 ours≤ref against truth) is the only sound adjudication; companion: fp16 max_abs=32 = 1 ULP at

- Former A5Ops ID: `OL-319`
- Fusion disposition: `additive_new`

`applies_to: soc=all (oracle/grader design, not silicon); cann=all; bisheng=n/a; op_class=any op verified against a vendor reference op`
`verified_on: 2_FFN bf16 inner_precise=0 + fp16 ip=1 cases (2026-07-25/26, Ascend910_9382, CANN 9.0.0)`

**Principle**: a vendor reference op is a *measurement instrument*, not ground truth. When the instrument contradicts its own spec sheet, any direct ours-vs-ref comparison measures the instrument's bug, not the kernel's. The adjudication that survives is a ladder: (T1) compare ours and ref each against an INDEPENDENT truth (CPU fp32 decomposition of the documented algorithm); (T2) require ours to be no worse than ref *against that truth* (`ours_err <= ref_err`). Each side is scored against truth separately; the broken instrument only ruins its own T1, not the verdict.

**Concrete anchor (2_FFN, bf16 `inner_precise=0`)**: the npu_ffn docstring says ip=0 means fp32 inner compute. Measured on 220x / CANN 9.0.0: ref-vs-CPU-truth MARE = 2e+1..1.2e+3 (the vendor op does something else, likely reduced-precision inner), while our kernel-vs-truth MARE = 6e-3..2e-1. An inline cand-vs-ref comparison FAILs these cases by construction — it would be grading against a reference that is itself 3 orders of magnitude off its own spec. Same shape on pseudo-quant (ref-vs-truth 1.3e+3).

**Companion aliasing hazard**: fp16 `max_abs_diff = 32` is 1 ULP at |y| ~ 32768 (fp16 ULP at 32768 is exactly 32) — a repeated identical max_abs=32 across cases is ULP-at-large-magnitude + small-value MARE aliasing, not a kernel bug. Check magnitude before treating a big absolute diff as a defect.

**Evidence**: 2_FFN (2026-07-25/26): bf16 ip=0 ref-vs-truth MARE 2e+1..1.2e+3 vs docstring spec; 8 fp16 cases with uniform max_abs=32 resolved as 1-ULP at |y|~32768 (MERE 1e-6..4e-5). Two-tier ladder adjudicated both correctly; inline comparison would have mis-failed both.

**Other instances (predicted)**: any op whose vendor reference has fused/precision-reduced internals that differ from its docstring (quant FFN variants, fused attention with inner_precise flags, fast-math activation epilogues). Rule of thumb: before trusting ours-vs-ref, run ref-vs-truth once per op class — if ref is off its own spec, switch the verdict to the ladder and say so in verification.json.
