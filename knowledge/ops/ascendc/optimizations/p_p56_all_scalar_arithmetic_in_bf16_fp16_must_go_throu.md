---
type: Optimization
title: "All \"scalar arithmetic\" in bf16/fp16 must go through the VEC path"
description: "Trigger: kernel has half/bf16 scalar arithmetic (not a tensor op). Pattern: NPU scalar FPU and Vector FPU are different hardware units. The same a / b done via scalar code vs Div(tensor) through the V"
tags: [task.optimization, topic.precision, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p56-all-scalar-arithmetic-in-bf16-fp16-must-go-throu.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# All "scalar arithmetic" in bf16/fp16 must go through the VEC path

> 原 OKF v1 卡号：**P-P56**。

## 优化点

**Trigger**: kernel has `half/bf16` scalar arithmetic (not a tensor op).

**Pattern**: NPU scalar FPU and Vector FPU are **different hardware units**. The same `a / b` done via scalar code vs `Div(tensor)` through the VEC pipeline may **produce different bits**.

**What CANN does**: all bf16/fp16 arithmetic goes through the VEC pipeline (tensor op + Duplicate to broadcast scalars).

**Kernel writing advice**:
- Need scalar `r = a op b` where a/b are bf16/fp16? → either cast to fp32 and compute at scalar level (safe), or construct a 1-element tensor and use a VEC op
- Do not assume `half x = half_y op half_z;` bit-matches the vector version
- Especially for `/` and `pow`: **always** go through VEC

**Impact on op #14**: my kernel has lots of half/bf16 scalar arithmetic (`pow_neg3_h = inv_std_h * inv_std_h * inv_std_h`, `grad_var_h = sum * -0.5 * pow`). These are all scalar paths and may not match CANN's vector path.

**Fix template**:
```cpp
// Old: scalar path
half inv_std_h = static_cast<half>(1.0f) / std_h;  // scalar FPU

// New: VEC path (at least for precision-sensitive scalars)
LocalTensor<half> smallBuf = scratchBuf_.Get<half>();  // allocate a small scratch buf
Duplicate(smallBuf, std_h, 16);  // VL-aligned length
Duplicate(onesBuf, static_cast<half>(1.0f), 16);
Div(smallBuf, onesBuf, smallBuf, 16);  // vector Div
half inv_std_h = smallBuf.GetValue(0);  // take the first element
```
Cost: extra VEC scratch buf + sync. Benefit: matches CANN's precision path.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（P-P56，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `precision`；trigger: Scalar and VEC units may round differently; compare both candidates against selected-source behavior and CPU fp64/backward truth.；优先级: HIGH。

> 索引行原文：`| P-P56 | All scalar arithmetic under bf16/fp16 must go through VEC path | precision | Scalar and VEC units may round differently; compare both candidates against selected-source behavior and CPU fp64/backward truth. | HIGH |`
