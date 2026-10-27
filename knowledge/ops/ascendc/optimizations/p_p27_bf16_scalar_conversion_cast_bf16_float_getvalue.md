---
type: Optimization
title: "bf16 scalar conversion — Cast(bf16→float) + GetValue"
description: "Key finding: bisheng does not support static_cast<float>(bfloat16_t) scalar conversion. SIMD Cast() vector intrinsic works fine. Anti-pattern: cpp // ❌ Compile fails: \"not support bf16 type cast\" bflo"
tags: [task.optimization, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.programming_model.execution_model.simt, topic.data_movement.data_copy, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p27-bf16-scalar-conversion-cast-bf16-float-getvalue.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# bf16 scalar conversion — Cast(bf16→float) + GetValue

> 原 OKF v1 卡号：**P-P27**。

## 优化点

**Key finding**: bisheng does not support `static_cast<float>(bfloat16_t)` scalar conversion. SIMD `Cast()` vector intrinsic works fine.

**Anti-pattern**:
```cpp
// ❌ Compile fails: "not support bf16 type cast"
bfloat16_t val = gmBuf.GetValue(i);
float fval = static_cast<float>(val);  // FAIL

// ❌ Lossy: bf16 exponent=8bit > half exponent=5bit → value range overflows to inf
Cast(halfBuf, bf16Buf, RoundMode::CAST_NONE, n);  // bf16→half is lossy!
```

**Correct pattern (P-P27)**:
```cpp
// ✅ bf16 scalar read: DataCopyPad → Cast(bf16→float) → GetValue(float)
DataCopyPad(bf16Buf, weightGm_[offset], copyParams, padNone);
PipeBarrier<PIPE_ALL>();
Cast(floatBuf, bf16Buf, RoundMode::CAST_NONE, count);  // bf16→float is lossless
PipeBarrier<PIPE_V>();
float w = floatBuf.GetValue(i);  // float scalar read works normally

// ✅ SIMT context (cannot use Cast): bit-manipulation workaround
float simt_to_float(bfloat16_t v) {
  uint16_t bits; __builtin_memcpy(&bits, &v, sizeof(bits));
  uint32_t f32 = (uint32_t)bits << 16;
  float r; __builtin_memcpy(&r, &f32, sizeof(r)); return r;
}
```

**Type conversion path table** (reg_convert.h):

| Source→Target | SIMD Cast() | Scalar static_cast | Note |
|---------|:-----------:|:---------------:|------|
| bf16→float | ✅ `asc_bfloat162float` | ❌ | **Use Cast then GetValue** |
| float→bf16 | ✅ `asc_float2bfloat16_rn` | ❌ | Cast then SetValue |
| bf16→half | ✅ `asc_bfloat162half_rn` | ❌ | **Lossy!** exponent overflow |
| half→bf16 | ✅ `asc_half2bfloat16_rn` | ❌ | Lossy (mantissa truncation) |
| half→float | ✅ `asc_half2float` | ✅ | Both work |
| float→half | ✅ `asc_float2half_rn` | ✅ | Both work |

**Decision rules**:
1. bf16 needs a scalar value → **Cast(bf16→float) first, then GetValue**; do NOT Cast(bf16→half)
2. bf16 scalar inside a SIMT kernel → **simt_to_float bit-manipulation** (SIMD Cast is unavailable)
3. half scalar conversion → `static_cast<float>(half)` works directly, no special handling needed

**Reference**: CANN `reg_convert.h`; minimal repro at `tests/repro/bf16_cast_repro.cpp`

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（P-P27，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `platform_compat`；trigger: bfloat16_t GetValue or static_cast；优先级: CRITICAL。

> 索引行原文：`| P-P27 | bf16 scalar via Cast(bf16→float) | platform_compat | bfloat16_t GetValue or static_cast | **CRITICAL** |`
