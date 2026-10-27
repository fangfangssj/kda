---
type: Optimization
title: "Fused dequant→activation→quant pipeline algorithm shape (generic for L2 fused activation+quant ops)"
description: "Generic algorithm structure for fused L2 ops of class dequant→activation→quant. Members: op#10 SwigluQuant, op#11 DequantSwigluQuant, future GroupNormSiluQuant / RMSnormGeluQuant / etc. Pipeline: (1)"
tags: [task.optimization, topic.implementation.pipeline, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p70-fused-dequant-activation-quant-pipeline-algorith.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3]
---

# Fused dequant→activation→quant pipeline algorithm shape (generic for L2 fused activation+quant ops)

> 原 OKF v1 卡号：**P-P70**。

## 优化点

Generic algorithm structure for fused L2 ops of class `dequant→activation→quant`. Members: op#10 SwigluQuant, op#11 DequantSwigluQuant, future GroupNormSiluQuant / RMSnormGeluQuant / etc. **Pipeline**: (1) Dequant: `Cast(x_fp, x_int32, NONE)` then `Mul(x_fp, x_fp, weight_scale)` then `Mul(x_fp, x_fp, activation_scale)` (broadcasted per-row); add bias if non-None. (2) Activation chunk + compute: see P-P71 for layout choice. (3) Smooth-quant: `Mul(out, out, quant_scale)` if quant_scale non-None. (4) Dynamic int8 quant tail: see P-P72. **UB layout for fused ops**: 4-6 buffers per row (x_int32 source, x_fp working, gate, linear, out_fp32, out_int8). For V220 (UB=192KB nominal, ~126KB effective per KC-2 candidate after runtime reservations) tile rows by D. **Per-row round-robin across AIV** — single-AIV-per-row, no inter-AIV reduction needed (quant amax is per-row). Determinism by-construction. Combine with P-P65 (cross-phase buffer aliasing) for tight UB budgets.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P70，convert_patterns_to_okf.py）。confidence 未升格。 -->
