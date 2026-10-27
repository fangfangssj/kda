---
type: Runbook
title: "Verify a hand-derived analytic backward against autograd on a PURE-fp64 forward — a reference forward that internally upcasts (.float()) injects a spurious ~1e-7 round-off that masks whether the derivation itself is correct"
description: "Verify a hand-derived analytic backward against autograd on a PURE-fp64 forward; a reference forward that upcasts with .float() injects ~1e-7 round-off that masks derivation correctness."
tags: [task.debugging, topic.precision, topic.implementation.masking]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-228-verify-analytic-backward-on-pure-fp64-forward.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
aliases: ["OL-228"]
---

# Verify a hand-derived analytic backward against autograd on a PURE-fp64 forward — a reference forward that internally upcasts (.float()) injects a spurious ~1e-7 round-off that masks whether the derivation itself is correct

> 原 OKF v1 卡号：**OL-228**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - An analytic backward shows a ~1e-7 gap vs torch.autograd and you cannot tell if the derivation is wrong — the reference forward upcasts internally with .float(), contaminating the comparison

## 现象 / 触发

Test-infra methodology (dtype-independent — autograd-vs-analytic comparison), all SoC / all CANN, hand-derived-backward derivation verification.

Verifying a hand-derived analytic backward has two **separable** questions: (1) is the *math derivation* correct, and (2) does the *kernel dtype path* hold precision. Conflating them wastes iterations.

The trap: a "reference" forward (e.g. a `forward_spec` whose job is to define the kernel algorithm) often **upcasts internally with `.float()`** for its own numerical stability. Running `torch.autograd` through it and comparing to your analytic derivation then shows a spurious ~1e-7 (fp32 round-off) gap — even when your derivation is bit-exact. That ~1e-7 is the forward's fp32 internal arithmetic, NOT a derivation bug, and it **masks** whether the derivation is actually right: you can neither confirm correctness (a real 1e-7 bug would hide under it) nor rule it out.

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训

To isolate question (1), compare your analytic gradient against `torch.autograd` on a **pure-fp64 forward** — a forward written entirely in float64 with no internal casts. The two should agree to fp64 round-off (err ≤ ~1e-15); any larger gap is a real derivation error, not noise.

**Decision rule**:
- To validate the **derivation**: build a dedicated pure-fp64 forward (no `.float()`, no dtype downcasts), run `torch.autograd.grad` on it, compare to your analytic formulas. Expect err ≤ ~1e-15. This is independent of the kernel and of test dtype.
- To validate the **kernel precision**: run the kernel in each target dtype vs the fp64 analytic oracle (per the Tier-1/Tier-2 thresholds, OL-103/OL-109). Keep this separate from the derivation check.
- Never use a `.float()`-upcasting reference forward as the derivation oracle — its internal round-off contaminates the comparison.

**Concrete anchor**:
```python
# DERIVATION check — pure fp64 forward, NO internal .float()
x64 = x.double().requires_grad_(True)
y64 = forward_fp64(x64, ...)              # written entirely in float64
(g_auto,) = torch.autograd.grad(y64.sum(), x64)
assert (g_auto - analytic_grad_fp64(x64, ...)).abs().max() < 1e-14   # bit-exact derivation
# WRONG: forward_spec(x.float()) -> autograd gap ~1e-7 (fp32 round-off) masks a 1e-7 derivation bug
```

**Evidence**: selective_scan_full_grad (2026-06-18, A5): the full 8-grad backward derivation (silu gate + D-skip + delta_bias added over the 5-grad sibling) was confirmed bit-exact (err ≤ 9e-16) only after switching the derivation oracle from `forward_spec` (which forces `.float()` internally, showing a spurious ~1e-7 autograd-vs-analytic gap) to a pure-fp64 forward. Cross-ref: OL-227 (the paired dtype-range classification once the derivation is confirmed fp64-bit-exact).
