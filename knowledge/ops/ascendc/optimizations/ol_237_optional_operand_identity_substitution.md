---
type: Optimization
title: "Optional-operand kernels collapse to ONE full-form template by substituting the operand's arithmetic-identity value, not by specializing per presence-combination"
description: "Feed an identity-valued buffer (zeros for an absent additive operand, ones for an absent multiplicative operand) to one full-form kernel; the substitution makes it bit-equivalent to every presence variant, eliminating N-1 template specializations."
tags: [task.optimization, topic.programming_model.execution_unit.vector]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-237-optional-operand-identity-substitution.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: [a3, "950"]
---

# Optional-operand kernels collapse to ONE full-form template by substituting the operand's arithmetic-identity value, not by specializing per presence-combination

> 原 OKF v1 卡号：**OL-237**。

## 优化点 / 选型

**When to consult:** an op has optional operands whose absence has a well-defined arithmetic identity (any optional-bias / optional-scale / optional-residual elementwise op; affine, layernorm-with-optional-affine, gated activations with optional gate; any FMA-shaped kernel). `applies_to: soc=all; cann=all; op_class=elementwise affine / optional-operand (hardware-independent algebraic identity)`.

### Principle

When an op has optional operands whose absence has a well-defined arithmetic identity — an additive operand absent => add 0, a multiplicative operand absent => multiply by 1 — author ONE full-form kernel and feed an identity-valued buffer (zeros / ones) for any absent operand, instead of emitting a separate kernel template per presence combination. The identity substitution makes the single full-form kernel **bit-equivalent** to every presence variant, eliminating N-1 template specializations and their dispatch. Validate **presence-balanced** (equal case counts per variant) so every branch is exercised.

Concrete anchor: DiT `modulate` `y = x*(1+scale)+shift` with optional scale/shift — upstream ships 3 templates (both / scale-only / shift-only). A zero buffer for absent scale (`1+0` => x1 identity) or absent shift (`+0` identity) makes one full-form kernel bit-equivalent to all three.

### Distinct from OL-154 (preempts a false conflict)

OL-154 keeps genuinely-different **per-strategy compute** (quantization modes) in separate files because no identity reduces one strategy to another. THIS rule fires ONLY when the variants differ by an operand reducible to an arithmetic identity element — same computation, just a neutral input. The two are complementary, not contradictory.

## 证据
- modulate kw-1 (2026-06-21, port_a3_to_a5 V220->arch35, A5 Ascend950PR_9579): one full-form VEC kernel + zeros-substitution = 225/225 PASS across presence-balanced 75/75/75 (both / scale-only / shift-only) x fp16/fp32/bf16. Saved 2 template specializations.
- Predicted: any optional-bias / optional-scale / optional-residual elementwise op; any FMA-shaped kernel where an absent term has a neutral element.
