---
type: Runbook
title: "The near-zero carve-outs are not a free pass — compare.py's small-value and cancellation carve-outs both end in passed = (error_count == 0); a carve-out changes the metric, not the slack"
description: "The near-zero carve-outs are not a free pass — compare.py's small-value and cancellation carve-outs both end in passed = (error_count == 0); a carve-out changes the metric, not the slack"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-403"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-403 — OL-403: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (grader logic, not silicon); cann=all; bisheng=n/a; op_class=all; phase=grading。
- 验证来源：read directly from src/scripts/orchestrator/precision/cannbench_grader/compare.py + thresholds.py, 2026-08-02。

**OL-403** — **the near-zero carve-outs are NOT a free pass**: `compare.py`'s small-value AND cancellation carve-outs BOTH end in `passed = (error_count == 0)` whenever the CPU competitor is clean. A carve-out changes the METRIC, not the slack. Two non-obvious consequences: (1) `cancel_mask` tests **our output** (`\

- Former A5Ops ID: `OL-403`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->
`applies_to: soc=all (grader logic, not silicon); cann=all; bisheng=n/a; op_class=all; phase=grading`
`verified_on: read directly from src/scripts/orchestrator/precision/cannbench_grader/compare.py + thresholds.py, 2026-08-02`

### Principle

The intuition **"tiny outputs get a carve-out, so near-zero error doesn't have to be accurate"
is false.** The 生态 `compare.py` grader has two carve-outs for near-zero golden values, and
BOTH of them end in the same fallback: *if the CPU reference produced no error at that
position, the NPU is allowed **zero** errors there.* A carve-out is a change of **metric**, not
a grant of **slack**. An algorithm that is relatively wrong in the near-zero band therefore
fails — it just fails through a different branch than a normal-region miss.

**The three regions a point can land in** (`|g|` = `|golden|`, `|o|` = our output):

| region | membership test | how it is graded |
|---|---|---|
| small-value | `\|g\| < SVT` | **absolute** err `> SV_ERR` counts; pass iff count `== 0` when CPU count `== 0`, else ratio `≤ 2` |
| cancellation | `\|o\| < CANCEL_ZERO` **AND** `SVT ≤ \|g\| < CANCEL_BOUNDARY` | rel err `> 10×threshold` counts; **same zero-tolerance fallback** |
| normal | everything else | rel err `> 10×threshold` ⇒ **immediate hard FAIL**, no fallback |

fp32 constants: `threshold = 2⁻¹³` (1.22e-4) ⇒ `mare_threshold = 10× = 1.22e-3`;
`SVT = 2⁻¹⁴` (6.10e-5); `SV_ERR = 2⁻³⁰`; `CANCEL_BOUNDARY = CANCEL_ZERO = 2⁻⁸` (3.91e-3).

**Two consequences that decide activation / tail-dominated ops:**

1. **The cancellation carve-out is conditioned on OUR output, not just golden.** `cancel_mask`
   requires `|output| < CANCEL_ZERO` as well as golden being in-band. A kernel that is wrong in
   the *large* direction (`|o| ≥ CANCEL_ZERO` while `|g| < CANCEL_ZERO`) is **not** in the
   cancellation region at all — it falls to **normal**, which is an unconditional FAIL. So the
   carve-out cannot be relied on before you know which way your error points.
2. **Landing inside a carve-out still requires being right.** With `cancel_cpu_error_count == 0`
   (the usual case — the CPU reference computes the band correctly), `cancel_passed` is
   literally `cancel_error_count == 0`. Same shape for small-value. There is no "N mismatches
   tolerated" budget.

### Concrete anchor

`src/scripts/orchestrator/precision/cannbench_grader/compare.py:579` — the membership test that
makes point (1) non-obvious, because it reads `output`, not only `golden`:

```python
output_near_zero      = output_abs < cancel_zero_threshold
golden_in_cancel_range = (golden_abs < cancel_boundary) & (golden_abs >= small_value_threshold)
cancel_mask = output_near_zero & golden_in_cancel_range & valid_mask
```

and the zero-tolerance fallback at `src/scripts/orchestrator/precision/cannbench_grader/compare.py:594-598` (`cancel_passed = (cancel_error_count == 0)`
when `cancel_cpu_error_count == 0`). Stage 1 (`src/scripts/orchestrator/precision/cannbench_grader/compare.py:489`) short-circuits to PASS on
overall MERE/MARE before any of this runs — so a kernel can also pass without touching a
carve-out at all, which is the outcome to aim for.

### Evidence

- celu kw-1 (2026-08-02, A5 Ascend950DT_9582, CANN 9.0.0, port_a3_to_a5): the naive
  `alpha1·(Exp(t)−1)` negative branch is ~100 % relatively wrong for `t ∈ [−0.01, 0)` — exactly
  the band this entry describes — yet still scored **156/156** by landing in the near-zero
  carve-out. The worker declined the passing-but-wrong form and shipped the identity-shaped
  `expm1` (OL-271 decision rule, P-P200). This is the entry's motivating case: **a green score
  obtained through a carve-out is not evidence the algorithm is right.**
- gelu_spec_grad kw-1 (2026-07-02, A5 Ascend950PR_957b): the converse, positive form — the
  erfc-direct rewrite passed Stage 1 outright with `small_value_total_count=0` and
  `cancel_total_count=0`, i.e. **zero carve-out reliance** (OL-271 anchor). Aim for this shape.

### Other instances (predicted)

Any op whose output legitimately approaches zero over part of its input domain: activation
negative branches (ELU/CELU/SELU/softplus/GELU tails), `1−cos`, `log1p`, residual/difference
outputs, gradient ops at stationary points, normalization near-zero cells, and any reduction
whose summands cancel. Also any agent reading `verification.json`: check
`small_value_total_count` / `cancel_total_count` — a large count means the verdict leaned on a
carve-out and deserves a second look at the algorithm.

**Cross-ref**: OL-271 (the numerical fix for the band this entry grades), P-P200 (constructive
recipe), OL-321 (a green whose scope does not cover the claim — same family), OL-103 (primitive
grade, the *other* axis; raising it does not help here).
