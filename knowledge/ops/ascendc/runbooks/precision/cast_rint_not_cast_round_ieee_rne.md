---
type: Runbook
title: "bf16/fp16 Cast default should be CAST_RINT, not CAST_ROUND (IEEE RNE)"
description: "bf16/fp16 Cast default should be CAST_RINT, not CAST_ROUND (IEEE RNE)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["RNE"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-81 — OL-81: bf16/fp16 Cast default should be CAST_RINT, not CAST_ROUND (IEEE RNE)

> 关联：同一 OL-81 主题已由既有卡 [ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md](ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：paradigm=ascendc（卡内 applies_to 仅声明范式；适用范围细节见正文）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行。

```yaml
applies_to:
  paradigm: ascendc
```
<!-- applies_to_backend: all -->
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass ops (17_EmbeddingWithInitialLayernormBackward, 20_FusedRopeWithQkNormAndKvCacheUpdate). Aligns with IEEE RNE which IS CPU-truth direction. Do not downgrade.
- **Category**: precision
- **Loaded by**: Generator (ALWAYS when using Cast for float type conversions), Pattern P-P50, P-P51
- **Trigger**: kernel uses `Cast(dst, src, RoundMode::CAST_ROUND, count)` for fp32→bf16 or fp32→fp16 and needs to bit-exactly match the torch_npu/PyTorch reference
- **Lesson**: AscendC's `CAST_ROUND` and `CAST_RINT` are **different** rounding modes:
  - `CAST_ROUND`: **round half UP** — if the first bit of the fraction being rounded is 1, round up (i.e. 0.5 always rounds up)
  - `CAST_RINT`: **IEEE 754 RNE** — round half to even (0.5 decided by parity of the last kept bit)
  - torch_npu / PyTorch / almost all IEEE 754 hardware defaults to **RNE**
  - **Therefore, to align with PyTorch bf16/fp16 behavior you must use CAST_RINT, not CAST_ROUND**
- **Empirical evidence**: op #14 AdaIN2D Backward: CAST_ROUND → 42/50 PASS (all bf16 cases fail); switching to CAST_RINT → 49/50 PASS. A single flag change fixed 8 bf16 cases.
- **Cast-mode-per-dtype split evidence** (fatrelu_mul port_a3_to_a5 kw-1, 2026-05-17): bf16 path uses CAST_RINT + fp16 path uses CAST_NONE (matching upstream V220 `FatreluMulND<bfloat16_t>::CopyOut` and `FatreluMulND<half>::CopyOut` respectively). Both achieve T1 bit-exact match to A3 ground truth. **The cast-mode-per-dtype split matters for bit-exact**: using CAST_RINT for FP16 would introduce 1-ULP drift on tie-breaks because upstream uses CAST_NONE for FP16. Rule: when porting from V220, mirror the upstream's per-dtype Cast mode rather than defaulting CAST_RINT for all float types.
- **Necessary-but-not-sufficient corollary** (ada_layer_norm Path A port, 2026-05-13): CAST_RINT corrects the final-write rounding mode, but it cannot rescue precision when the dominant divergence is an upstream **dtype-cast chain mismatch** (kernel keeps fp32 across a stage where the reference casts to bf16/fp16 mid-pipeline; see P-P50). Iter-2 of ada_layer_norm applied CAST_RINT alone and saw no MARE improvement because the actual bug was post-mod arithmetic happening in fp32 vs native dtype. **Workflow rule**: when the rounding-mode fix doesn't move MARE, do NOT escalate to "try a different rounding mode" — read the reference for intermediate `.to(native)` calls and apply P-P50.
- **Detection method**: grep the kernel for `Cast(` and `RoundMode::`. If you see `CAST_ROUND` used for fp32→bf16/fp16 rounding, **it almost always must be changed to `CAST_RINT`**. The only exception: when the source algorithm explicitly requires round-half-up (extremely rare, usually not the case).
- **Doc source**: hiascend.com AscendC Cast description, query via `python3 src/scripts/fetch_ascendc_doc.py Cast`
- **Related**: P-P50 (bf16 intermediate state), P-P51 (scalar chain native dtype), OL-80 (look up API before writing)
- **Evidence**: E2 level (empirical: a single flag flip raised bf16 from 1/9 PASS to 8/9 PASS)
