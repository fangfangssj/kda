---
type: Optimization
title: "Reciprocal via tensor Div(1, x), not scalar `1/x`"
description: "Trigger: kernel needs to compute 1/x or c/x (c is a scalar constant). Pattern: CANN aclnnReciprocal does not use Newton-Raphson nor a hardware reciprocal instruction — it uses a direct tensor-level Di"
tags: [task.optimization, topic.precision, topic.programming_model.execution_unit.vector, topic.implementation.masking]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p54-reciprocal-via-tensor-div-1-x-not-scalar-1-x.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Reciprocal via tensor Div(1, x), not scalar `1/x`

> 原 OKF v1 卡号：**P-P54**。

## 优化点

**Trigger**: kernel needs to compute `1/x` or `c/x` (c is a scalar constant).

**Pattern**: CANN `aclnnReciprocal` does not use Newton-Raphson nor a hardware reciprocal instruction — it uses a direct tensor-level Div:
```cpp
// reciprocal_dag.h core loop
AscendC::MicroAPI::Duplicate(ones, (T)1.0, mask);       // build constant tensor of 1.0
AscendC::MicroAPI::Div<T>(vregOutput, ones, vregInput);  // elementwise 1 / x
```

**Requirements**:
- Do not write `half inv_x = static_cast<half>(1.0f) / x_h;` (scalar FPU path — not guaranteed bit-match with vector Div).
- Broadcast x to a tensor and use VEC `Div(ones_tensor, x_tensor)`.
- Or, if only a single scalar reciprocal is needed, Duplicate to a small tensor, Div, then GetValue.

**When it matters**: when the kernel's internal scalar chain passes through multiple reciprocal/div ops (e.g., `pow(std, -3) = 1/std * 1/std * 1/std`), scalar vs vector precision differences accumulate.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（P-P54，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `precision`；trigger: Select scalar or vector form from OL-97 selected-source behavior plus CPU fp64 edge probes. Target CANN bodies are advisory prior-art.；优先级: HIGH。

> 索引行原文：`| P-P54 | Reciprocal uses tensor Div(1,x), not scalar 1/x | precision | Select scalar or vector form from OL-97 selected-source behavior plus CPU fp64 edge probes. Target CANN bodies are advisory prior-art. | HIGH |`
