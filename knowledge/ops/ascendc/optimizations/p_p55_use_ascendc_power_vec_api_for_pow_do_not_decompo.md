---
type: Optimization
title: "Use AscendC `Power` VEC API for Pow; do not decompose manually"
description: "Trigger: computing pow(x, y) or pow(x, const_int). Pattern: CANN has a dedicated Power<T> VEC template per dtype: cpp // pow_bf16_nddma_without_loops.h Power<bfloat16_t, false, pConfig_>(dstBuf, baseB"
tags: [task.optimization, topic.precision, topic.programming_model.execution_unit.vector]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p55-use-ascendc-power-vec-api-for-pow-do-not-decompo.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Use AscendC `Power` VEC API for Pow; do not decompose manually

> 原 OKF v1 卡号：**P-P55**。

## 优化点

**Trigger**: computing `pow(x, y)` or `pow(x, const_int)`.

**Pattern**: CANN has a dedicated `Power<T>` VEC template per dtype:
```cpp
// pow_bf16_nddma_without_loops.h
Power<bfloat16_t, false, pConfig_>(dstBuf, baseBuf, expBuf, count);
```
`Power`'s internal implementation (fused bf16/fp16/fp32 paths) differs completely from a manual `1/x * 1/x * 1/x`, with different rounding paths.

**Counter-example** (my kernel for `pow(std, -3)`):
```cpp
half inv_std = 1.0f / std;
half m3 = inv_std * inv_std * inv_std;  // WRONG: 3 muls, not bit-equal to Power(std, -3)
```

**Correct** (use the public AscendC VEC API):
```cpp
// broadcast std to a tensor
LocalTensor<half> stdT = ...; Duplicate(stdT, std_h, count);
LocalTensor<half> expT = ...; Duplicate(expT, static_cast<half>(-3.0), count);
Pow(resT, stdT, expT, count);  // OK: vector-level pow, matches CANN
// GetValue(resT, 0) if a scalar is needed
```

**When it matters**: whenever the reference contains `torch.pow(x, y)` or `x ** y` and bit-exact matching is required.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（P-P55，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `precision`；trigger: Decide among advisory target-derived candidates using selected-source capture and CPU fp64 edge/backward probes.；优先级: HIGH。

> 索引行原文：`| P-P55 | Pow strategy depends on exponent: integer/half exponents use Mul/Div/Sqrt decomposition, general exponent uses Power API | precision | Decide among advisory target-derived candidates using selected-source capture and CPU fp64 edge/backward probes. | HIGH |`
