---
type: Runbook
title: "Edge-case discovery methodology (quantization ops)"
description: "Edge-case discovery methodology (quantization ops)"
tags: [task.debugging, topic.data_movement, topic.memory, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-29-edge-case-discovery-methodology-quantization-ops.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-29"]
---

# Edge-case discovery methodology (quantization ops)

> 原 OKF v1 卡号：**OL-29**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - any op involving bit manipulation, quantization, or fixed-point arithmetic

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process, algorithm_selection
- **Loaded by**: Analyzer, QA
- **Trigger**: any op involving bit manipulation, quantization, or fixed-point arithmetic
- **Lesson**: During MXFP4 migration, several classes of edge cases were found:
  1. **Shared exponent extreme-value divergence**: having 1e+38 and 1.0 in the same group causes 1.0 to underflow to 0 (correct behavior but needs verification)
  2. **Integer overflow**: `1 << expdiff` when expdiff > 30 is C++ UB (source produces 0, CPU produces inf)
  3. **Rounding boundary**: behavior of round-to-nearest-even at 0.5 differs between implementations (source bit ops vs PyTorch floor+0.5)

  **Edge-case discovery method**:
  - Extreme-value combinations: mix extreme and normal values in the same group/block
  - Integer overflow: find all `<<` and `>>` ops, check the upper bound on shift amount
  - Rounding boundary: construct values exactly on the quantization boundary (e.g. MXFP4's 1.5 × 2^exp boundary)
  - Zero and subnormals: 0.0, -0.0, smallest positive subnormal
  - Saturation: max representable value + 1 ULP
  - Three-way comparison: PyTorch (spec) vs source vs AscendC; any pair disagreeing is a bug
- **Evidence**: MXFP4 session (2026-04-07)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-29（category=process, algorithm_selection，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
