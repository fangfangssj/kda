---
type: Runbook
title: "A fused CANN reference op's bit-level semantics differ across SoC — hardware-probe the target NPU, never transplant a sibling-SoC semantic table"
description: "A fused CANN reference op's bit-level semantics differ across SoC — hardware-probe the target NPU, never transplant a sibling-SoC semantic table"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-480"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-480 — OL-480: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (cross-SoC ports of fused-CANN-reference ops); cann=all; bisheng=n/a; op_class=quant-matmul family and any op judged against a fused vendor reference。
- 验证来源：6_QuantMatmul a3 port (2026-08-01, Ascend910_9382 vs sibling a5 tables, CANN 9.0.0)。

**OL-480** — a fused CANN reference op's bit-level semantics (multiply order, scale crop, bias-add position, silently-ignored args) differ across SoC; hardware-probe the target NPU, never transplant a sibling-SoC semantic table. Anchor: `aclnnQuantMatmulV5` 910C pertoken `(acc*scale)*pt` vs a5 table `(acc*pt)*scale`.

- Former A5Ops ID: `OL-480`
- Fusion disposition: `additive_new`

`applies_to: soc=all (cross-SoC ports of fused-CANN-reference ops); cann=all; bisheng=n/a; op_class=quant-matmul family and any op judged against a fused vendor reference`
`verified_on: 6_QuantMatmul a3 port (2026-08-01, Ascend910_9382 vs sibling a5 tables, CANN 9.0.0)`

**Principle**: a fused vendor op's exact arithmetic — operand multiply order, where the scale is applied/cropped, where bias is added, and which arguments are silently IGNORED — is a per-SoC property, not a family property. Importing the sibling chip's semantic table as ground truth grades the kernel against the wrong algorithm. Probe the target NPU's behavior on-device (small hand-driven cases over the argument matrix) and pin the semantics per chip before any bit-level comparison work.

**Concrete anchor**: `aclnnQuantMatmulV5` — 910C pertoken order is `(acc*scale)*pt`, while the sibling a5 CPU-truth table says `(acc*pt)*scale`. Same op name, different arithmetic.

**Evidence**: 6_QuantMatmul a3 port (2026-08-01, worker on-device probe of the vendor semantics before kernel numerics).

**Other instances (predicted)**: WeightQuantBatchmatmul / QuantMatmulReduceSum / GroupedMatmulSwigluQuant family (same fused-vendor-reference class); any op with "ignored by the reference" arguments discovered only by probing.
