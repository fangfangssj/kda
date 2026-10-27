---
type: Runbook
title: "Im2Col + Matmul decomposition for Conv2D — sliding-window ops via spatial-to-column transform + Cube matmul"
description: "Im2Col + Matmul decomposition for Conv2D — sliding-window ops via spatial-to-column transform + Cube matmul"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-FA-CV-10"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-FA-CV-10 — Im2Col + Matmul decomposition for Conv2D — sliding-window ops via spatial-to-column transform + Cube matmul

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；适用条件见正文 Applicability 段（任何空间维滑窗 pattern 的算子——Conv2D/Conv3D/Pooling/DilatedConv/DepthwiseConv）。
- 验证来源：cv-agent conv2d kernel 架构（im2col GM→L1 tile + Cube Matmul）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`verified_on: cv-agent conv2d kernel architecture (im2col GM→L1 tile + Cube Matmul)`

**Pattern**: Decompose Conv2D (and any sliding-window op) into two stages: (1) Im2Col — DataCopy transforms spatial window into column matrix in L1, (2) Matmul — Cube multiplies column matrix × filter matrix. The decomposition converts the irregular spatial access pattern into a regular matmul that Cube can execute at full throughput.

**Im2Col tile sizing** (critical for L1 budget):
```
L1 input tile rows = (KH - 1) * dilationH + (FMAP_L1_TILE_HO - 1) * strideH + 1
L1 input tile cols = (KW - 1) * dilationW + (FMAP_L1_TILE_WO - 1) * strideW + 1
```
The L1 tile must be large enough to cover one output tile worth of input windows — each output position needs KH×KW input elements.

**Why not direct convolution**: Direct spatial sliding-window access to GM is stride-irregular → cannot use VEC/Cube efficiently. Im2Col transforms it into a dense matmul → Cube runs at full throughput. The im2col overhead (one extra DataCopy pass) is amortized by Cube matmul speed.

**Applicability**: Any op with a sliding-window pattern over spatial dimensions — Conv2D, Conv3D, MaxPool, AvgPool, DilatedConv, DepthwiseConv. The pattern generalizes to N-D by adjusting the im2col tile size formula.

**Detection**: grep for `Conv2D\|Im2Col\|im2col` in kernel .h. If kernel directly loops over spatial positions with scalar access → gap. If kernel has im2col tile sizing + Cube Matmul → pattern applied.

**Evidence**: cv-agent `conv2d/kernel/conv2d_kernel.h` — `Conv2DCubeKernel` class with `hiBlock_`/`wiBlock_` im2col tile sizing + Cube Matmul via L0A/L0B queues with fp32 accumulation.

**Cross-ref**: CAND-FA-CV-8 (fused matmul — conv2d is structurally "im2col fused with matmul", same L1-resident intermediate principle), CAND-FA-CV-7 (multi-strategy dispatch — pooling ops may need im2col variant vs direct variant per kernel size)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-10，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
