---
type: Runbook
title: "`static_cast<float>(bfloat16_t)` fails in bisheng"
description: "`static_cast<float>(bfloat16_t)` fails in bisheng"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-5-static-cast-float-bfloat16-t-fails-in-bisheng.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-5"]
---

# `static_cast<float>(bfloat16_t)` fails in bisheng

> 原 OKF v1 卡号：**EC-5**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: not support bf16 type cast
  ```
  or:
  ```
  error: static_cast from 'bfloat16_t' to 'float' is not allowed
  ```
- **Root cause**: Bisheng compiler (CANN 9.0.0 and 9.0.T501) does not support scalar `static_cast` between `bfloat16_t` and `float` in either direction. The `half` (fp16) type works fine with `static_cast`. This is a known bisheng limitation (PB-4 in PLATFORM_BUGS.md).
- **Fix (SIMT kernel — use bit-manipulation)**:
  ```cpp
  // BEFORE (fails):
  bfloat16_t val = input[i];
  float fval = static_cast<float>(val);    // ❌ bisheng rejects this

  // AFTER (bit-manipulation workaround):
  template <typename T>
  __aicore__ inline float simt_to_float(T v) { return static_cast<float>(v); }

  template <>
  __aicore__ inline float simt_to_float<bfloat16_t>(bfloat16_t v) {
    uint16_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    uint32_t f32bits = static_cast<uint32_t>(bits) << 16;
    float result;
    __builtin_memcpy(&result, &f32bits, sizeof(result));
    return result;
  }

  // Reverse: float → bfloat16_t
  template <typename T>
  __aicore__ inline T simt_from_float(float v) { return static_cast<T>(v); }

  template <>
  __aicore__ inline bfloat16_t simt_from_float<bfloat16_t>(float v) {
    uint32_t f32bits;
    __builtin_memcpy(&f32bits, &v, sizeof(f32bits));
    uint16_t bits = static_cast<uint16_t>(f32bits >> 16);  // truncate
    bfloat16_t result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
  }
  ```
- **Fix (SIMD kernel — use Cast intrinsic)**:
  ```cpp
  // Cast(bf16→float) is lossless and works:
  Cast(floatBuf, bf16Buf, RoundMode::CAST_NONE, count);
  float w = floatBuf.GetValue(i);
  ```
- **WARNING**: `Cast(bf16→half)` is LOSSY — bf16 exponent=8bit overflows half exponent=5bit, producing `inf` for large values. Always cast bf16→float (lossless).
- **Related**: P-P27 (bf16 scalar via Cast + GetValue)

<!-- 迁移自 porter kb/target/ascendc/（EC-5，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
