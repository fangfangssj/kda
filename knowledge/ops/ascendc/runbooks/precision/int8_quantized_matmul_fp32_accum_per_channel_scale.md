---
type: Runbook
title: "INT8 quantized matmul with FP32 accumulation + per-channel scale → FP16 output"
description: "INT8 quantized matmul with FP32 accumulation + per-channel scale → FP16 output"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-FA-CV-11"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-FA-CV-11 — INT8 quantized matmul with FP32 accumulation + per-channel scale → FP16 output

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；适用条件见正文（INT8 量化 matmul，fp32 累加 + per-channel scale → fp16 输出；K ≤ 4096 时累加无噪声的精度论证见正文 Precision guarantee 段）。
- 验证来源：cv-agent quant_matmul kernel（int8 A × int8 B → fp32 accum → scale → fp16）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`verified_on: cv-agent quant_matmul kernel (int8 A × int8 B → fp32 accum → scale → fp16)`

**Pattern**: For INT8 quantized matrix multiplication, compute in fp32 accumulation (exact for INT8 range), apply per-channel scale, then convert to fp16 output. All in one kernel — no intermediate GM tensors.

**Mechanism**:
1. Load INT8 A, INT8 B from GM into L1
2. Cube Matmul: INT8 × INT8 → INT32 in L0C
3. Convert INT32 → FP32 (exact for |result| ≤ 2^24)
4. Multiply per-channel scale (FP32) elementwise
5. Convert FP32 → FP16 for output
6. Write FP16 output to GM

**Why not INT8 output**: INT8 output requires requantization with zero-point — loses precision and makes the kernel specific to one quantization scheme. FP16 output is universal — downstream ops consume it without knowing the quantized origin. The INT8→FP16 conversion is zero-cost on NPU (type cast in VEC).

**Precision guarantee**: INT8 range is [-128, 127], so max |INT8 product| = 16384 per element pair. With K ≤ 4096, max |sum| = 67,108,864 < 2^26 — fits in FP32 exact-representation range (2^24 for integers). No accumulation noise for K ≤ 4096.

**Detection**: grep for `int8\|INT8\|int8_t` AND `Matmul\|Cube` in kernel .cpp. If activation is quantized but weights are fp16 → partial quantization, not this pattern.

**Evidence**: cv-agent `quant_matmul/` with model.py forward: `a.to(fp32) @ b.to(fp32) * scale → fp16`. Kernel pattern confirmed via matmul_leakyrelu's Matmul + elementwise pipeline (CAND-FA-CV-8).

**Cross-ref**: CAND-FA-CV-8 (fused matmul+elementwise — quant matmul is "matmul + scale multiply" fusion, same L1-resident principle), P-P93 (quant-op CPU reference `.clamp(low,high).to(int_dtype)`), P-P94 (MERE/MARE aux precision standard)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-11，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
