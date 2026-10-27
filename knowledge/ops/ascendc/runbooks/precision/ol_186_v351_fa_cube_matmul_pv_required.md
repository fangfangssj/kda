---
type: Runbook
title: "V351 forward FA — cube-MatmulImpl P@V is required to match CANN precision; vec-only scalar-broadcast P@V drifts +5-25pp mare at large n_kv_tile"
description: "CANN V351 FA runs P·V on the cube unit with an fp32 fused-MAC accumulator; a vec-only scalar-broadcast P@V emulation drifts +5-25pp mare at large n_kv_tile — matching it is a structural cube choice."
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.implementation.atomic, topic.implementation.masking]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-186-v351-fa-cube-matmul-pv-required.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
aliases: ["OL-186"]
---

# V351 forward FA — cube-MatmulImpl P@V is required to match CANN precision; vec-only scalar-broadcast P@V drifts +5-25pp mare at large n_kv_tile

> 原 OKF v1 卡号：**OL-186**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - a vec-only row-tiled FA kernel scores far below CANN (e.g. 12/61) with a +5-25pp mare gap that grows with n_kv_tile

## 现象 / 触发
A vec-only row-tile FlashAttention kernel that emulates P@V with a software scalar-broadcast loop passes only a small fraction of cases vs CANN (12/61 baseline) and shows a mare gap of +5–25pp that scales with `n_kv_tile`.

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
CANN's V351 FA is a **two-block cooperative kernel**:
- **Cube block** (`flash_attention_score_block_cube.h`) does BOTH `Bmm1: Q·Kᵀ→S` AND `Bmm2: P·V→O` via `MatmulImpl<...>`. The cube hardware computes `acc[i,j] += Σ_k a[i,k]*b[k,j]` atomically per cube instruction in an **internal fp32 fused-multiply-add accumulator**, with no per-inner-element rounding.
- **Vec block** (`flash_attention_score_block_vec_infer.h`) does softmax (max, sub-max, exp, sum), Cast fp32→out dtype, and output assembly. `Bmm2FDOut` (line 697) is **result handling** (DataCopy / DataCopyPad to GM), NOT the P@V compute.

A vec-only emulation pays SIMD-Mul + SIMD-Add per `(kk)` pair with software fp32 rounding at every step; at `T_kv × n_kv_tile = 512` sequential fp32 FMAs the drift accumulates to the observed +5–25pp mare gap. **Matching CANN precision is a STRUCTURAL choice (do P@V on the cube), not a summation-order tweak.**

**Measurement matrix (PR #139, 2026-05-24):**

| Technique | PASS (T1+T2) | Notes |
|---|---|---|
| Baseline sequential scalar-broadcast P@V | 12/61 | naive `oF[i,:] += V[kk,:] * p_ik` chained-add |
| Kahan compensated summation | 13/61 | flips only case 52; cases 1/2/5/9 unmoved |
| Tree-reduce (binary fold) | 13/61 | flips case 5 (gap +11.88pp→0 bit-exact); cases 1/2/9 unmoved |
| Mask + sink + sentinel fixes | 42/61 | independent of P@V drift — unlocks pse/atten_mask/sink/sparse_mode coverage |

**Critical caveat**: the chained-depth drift model only PARTIALLY predicts which cases close — layout-specific behavior seen (SBH case 5 closes with tree-reduce, BSH case 1 of the same shape does not), suggesting a memory-order / DataCopy-stride interaction beyond pure rounding drift. The dominant gap for `n_kv_tile≥4` is the **structural cube-vs-vec architecture difference**, not drift alone.

**Mare-gap pattern (vec-only baseline)**: case 52 `[1,8,128,64]` BNSD n_kv_tile=4 D=64 → +0.00pp (bit-tied); case 9 `[1,128,8,128]` BSND n_kv_tile=4 D=128 → +5.47pp; case 1 `[1,256,2048]` BSH n_kv_tile=8 D=128 → +9.23pp.

Verified on soc=Ascend950PR_9579 (V351), cann=9.0.0. Source: CANN grep `ops-transformer/attention/common/op_kernel/arch35/` (2026-05-23).
