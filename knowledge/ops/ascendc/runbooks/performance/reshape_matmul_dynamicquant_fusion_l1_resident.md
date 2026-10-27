---
type: Runbook
title: "Reshape + Matmul + DynamicQuant fusion — multi-stage pipeline with L1-resident intermediates"
description: "Reshape + Matmul + DynamicQuant fusion — multi-stage pipeline with L1-resident intermediates"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-FA-CV-13"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-FA-CV-13 — Reshape + Matmul + DynamicQuant fusion — multi-stage pipeline with L1-resident intermediates

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；适用条件见正文（reshape+matmul+dynamic-quant 可融合的算子；L1 预算约束 K ≤ L1_size/(element_size×2)，超出需分 tile）。
- 验证来源：cv-agent reshape_matmul_rowwise_quant_int8 kernel（reshape view → mm → dynamic_quant 单 kernel）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`verified_on: cv-agent reshape_matmul_rowwise_quant_int8 kernel (reshape view → mm → dynamic_quant, all in one kernel)`

**Pattern**: Fuse reshape (zero-copy view), matmul, and dynamic quantization into a single AscendC kernel. The reshape is free (reinterpret strides), the matmul output stays in L1, and the VEC quant stage reads directly from L1. Two intermediates (reshaped view, matmul result) never touch GM.

**Pipeline**:
1. Reshape: x(m,n) → x_view(m*n/k, k) — zero-copy, just reinterpret strides
2. Matmul: x_view @ h(k,k) → result(m*n/k, k) in L1
3. DynamicQuant: row-wise max_abs → scale → round + clip → int8 output
4. Reshape back: int8 result → (m, n) output

**Why fusion matters**: Without fusion, the matmul output (fp16/bf16, size m*n) must be written to GM, then read back by a separate quant kernel. For m=4096, n=2048: 16MB write + 16MB read avoided.

**L1 budget constraint**: The matmul output buffer + quant workspace must fit in L1 simultaneously. This constrains the inner dimension K to K_max = L1_size / (element_size * 2). If K exceeds this, the pipeline must split into tiles.

**Detection**: grep for `reshape` AND `Matmul\|GEMM` in the same kernel .cpp. If reshape+matmul are separate ops → gap. If in same kernel → fusion applied.

**Evidence**: cv-agent `reshape_matmul_quant/` kernel with `ReshapeMatmulQuantKernel` class — single Process() body with reshape→matmul→quant stages.

**Cross-ref**: CAND-FA-CV-8 (fused matmul+elementwise — same principle, different tail stage: quant vs activation), CAND-FA-CV-11 (INT8 quant matmul — this pattern adds reshape fusion before matmul), CAND-FA-CV-1 (ring buffer with WorkspaceQueue — L1 workspace management generalizes to reshape+matmul+quant pipeline)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-13，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
