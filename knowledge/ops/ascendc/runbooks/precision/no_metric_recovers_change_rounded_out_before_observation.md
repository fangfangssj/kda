---
type: Runbook
title: "No metric can recover a change rounded out before observation — choose the observation point first"
description: "No metric can recover a change rounded out before observation — choose the observation point first"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-441"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-441 — OL-441: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; bisheng=n/a; op_class=all (reduced-precision outputs and multi-limb representations)。
- 验证来源：executable host arithmetic counterexample, torch bf16, 2026-08-16。

**OL-441** — **no metric can recover a change rounded out before observation.** At unit magnitude bf16 ULP is `0.0078125`; `1 + 2^-12` stores as exactly `1.0`, so both `max_abs` and nRMSE are zero. Choose the observation point first: retain a pre-cast accumulator for sub-ULP diagnosis, or accept byte-identical stored outputs at the product layer. Score reconstructed multi-limb quantities, not an isolated residual limb. Cross-ref [OL-297](../validations/check_discriminating_power_defect_absent_vs_present.md), [OL-305](../validations/control_baseline_reference_path_themselves_under_test.md), PRECISION_STANDARD_v2.1.md. backend=all.

- Former A5Ops ID: `OL-441`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (reduced-precision outputs and multi-limb representations)`
`verified_on: executable host arithmetic counterexample, torch bf16, 2026-08-16`

**Principle.** Quantisation happens before a metric reads a stored tensor. If two pre-rounding values map to the
same stored value, every deterministic metric over those stored tensors — `max_abs`, nRMSE, ULP distance, or a
per-element tolerance count — sees identical inputs and returns zero difference. Changing the metric cannot
recover information that the representation already discarded. Conversely, `max_abs` is **not capped** at one
ULP: its smallest non-zero step at a given magnitude is one ULP, but it can span arbitrarily many ULPs.

**Executable counterexample.** At unit magnitude bf16's next representable value is `1.0078125`, so its ULP is
`0.0078125 = 2^-7`. The pre-rounding perturbation `1 + 2^-12` stores as exactly `1.0`:

```python
import torch

x = torch.tensor([1.0], dtype=torch.bfloat16)
y = torch.tensor([1.0 + 2.0**-12], dtype=torch.bfloat16)
assert torch.equal(x, y)
error = x.float() - y.float()
nrmse = error.square().mean().sqrt() / x.float().square().mean().sqrt()
assert error.abs().max().item() == 0.0
assert nrmse.item() == 0.0
```

**Rule.** Decide which quantity the product contract makes observable. To diagnose a sub-ULP algorithmic
change, retain and compare a pre-cast accumulator or a higher-precision oracle; if the contract is the stored
bf16 output, byte-identical bf16 tensors are genuinely identical at that layer. For a multi-limb representation
such as `hi + lo`, score the **reconstructed quantity** used by the consumer, not either limb in isolation; a
relative error on a near-zero residual answers a different question from reconstruction accuracy.

- **Evidence**: the counterexample above was executed on 2026-08-16 and produced `x=y=1.0`, `max_abs=0`,
  `nRMSE=0`, and nextafter ULP `0.0078125`.
- **Other instances (predicted)**: fp16/bf16/fp8 outputs, int8 dequantisation, compensated sums, and split
  operands. The transferable test is whether the compared bytes still contain the information the metric is
  expected to reveal.
- **Cross-reference**: OL-297 (a check needs distinct defect-present/absent outputs), OL-305 (the apparatus is
  under test), `PRECISION_STANDARD_v2.1.md`.
