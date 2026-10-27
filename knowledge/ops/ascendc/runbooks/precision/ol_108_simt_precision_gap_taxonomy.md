---
type: Runbook
title: "SIMT precision-gap taxonomy — distinct from OL-102 chain rounding"
description: "OL-102 chain-rounding fixes do NOT generalize to SIMT-class ops; SIMT precision gaps split into five distinct classes, each needing a different fix path."
tags: [task.debugging, topic.precision, topic.programming_model.execution_model.simt, topic.implementation.atomic]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-108-simt-precision-gap-taxonomy.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
aliases: ["OL-108"]
---

# SIMT precision-gap taxonomy — distinct from OL-102 chain rounding

> 原 OKF v1 卡号：**OL-108**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - SIMT-class op (gather / scatter / index_put / interp / conv / pool / embedding) is PARTIAL on the CPU-truth precision standard

## 现象 / 触发
A SIMT-class op (gather / scatter / index_put / interpolate / conv / pool / embedding) is PARTIAL against the CPU-truth standard. The instinct is to reach for OL-102 (chain rounding: "kernel needs per-op T to match the CPU per-op T reference"). Validated from a 4-op SIMT survey (op#28 Interpolate, op#26 AvgPool3d, op#19 IndexPut, op#21 Scatter) testing OL-102 generalization.

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
OL-102 is the dominant precision-gap class for SIMD-style **composite Mul/Add ops** (Rope-mul, MoE-finalize, weighted-sum), but it does NOT generalize to SIMT-class ops. SIMT precision gaps fall into distinct classes, each with its own fix:

| Class | Symptom | Examples | Fix path |
|---|---|---|---|
| **Atomic-write determinism** | run-to-run jitter on duplicate/conflicting indices; ours==CANN both fail CPU-truth bit-exact | scatter(reduce='add'), index_put accumulate=True, embedding-bwd | NPU torch reference is non-deterministic while CPU is deterministic. Either serialize by construction (single-core-per-row), or accept a best-effort precision policy. |
| **Parallel reduction order** | sum/mean reductions diverge from CPU sequential order (non-associative fp) | avg_pool3d, fp16 sum-reductions over parallel cores | Hard to fix portably; CPU-truth bit-exact may be unrealistic. Tier-2 acceptable. |
| **Reference-upcasts-to-fp32** | V0 (fp32 internal) is correct; V_ALL (per-op T) hurts | interpolate (bilinear/bicubic), conv2d/3d, pool variants | Leave V0 alone — OL-102 NOT applicable. Read PyTorch ATen source to confirm the upcast. |
| **MARE-noise at near-zero outputs** | MERE small (sub-fp32-ULP) but MARE huge (>10x threshold) | bicubic on smooth images (cubic weights nearly cancel) | eval-tool false-positive: the `MARE < 10 x thr` formula over-penalizes near-zero outputs. Consider tightening eval epsilon or an absolute-tolerance fallback for tiny \|golden\|. |
| **SIMT scalar bf16 conversion drift** | bf16 fails worse than fp16 on the same op; `simt_to_float<bfloat16_t>` / `simt_from_float<bfloat16_t>` bit-tricks may have subtle ordering diff vs PyTorch CPU | op#28 bilinear bf16 cases 65/66 (3 bf16 ULP off) | verify the bit-trick matches `c10::BFloat16::round_to_nearest_even`; check whether PyTorch CPU does non-RNE conversions in specific paths |

### Decision flow for SIMT op precision gaps
1. Read `model.py`: does CPU explicitly upcast (`.float()` then compute then `.to(T)`)? If YES → V0 is correct; the gap is NOT chain rounding, skip OL-102 entirely. If NO → continue.
2. Run a Stage-1 PyTorch probe (V0 fp32-chain vs V_ALL per-op T). If V0 is closer to CPU, treat the fp32-internal path as the reference (rather than assuming per-op-T alignment). Classify the residual into the table above.
