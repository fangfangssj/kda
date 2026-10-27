---
type: Optimization
title: "fp16/bf16 Scalar Kernel Argument Passing"
description: "Problem: extern \"C\" __global__ __aicore__ kernel entry cannot directly take half/bfloat16_t scalar parameters. The ABI does not support this and it causes value corruption or undefined behaviour. Anti"
tags: [task.optimization, topic.implementation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p30-fp16-bf16-scalar-kernel-argument-passing.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# fp16/bf16 Scalar Kernel Argument Passing

> 原 OKF v1 卡号：**P-P30**。

## 优化点

**Problem**: `extern "C" __global__ __aicore__` kernel entry cannot directly take `half`/`bfloat16_t` scalar parameters. The ABI does not support this and it causes value corruption or undefined behaviour.

**Anti-pattern**:
```cpp
extern "C" __global__ __aicore__ void init_kernel_fp16(
    GM_ADDR data, half num, int64_t size) {  // ❌ half cannot cross the extern "C" boundary
```

**Correct pattern** (uint16_t bit-pattern):
```cpp
extern "C" __global__ __aicore__ void init_kernel_fp16(
    GM_ADDR data, uint16_t num_bits, int64_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  half num;
  *reinterpret_cast<uint16_t*>(&num) = num_bits;  // rebuild from bit pattern
  // ... use num ...
}

// Same idea for bf16:
extern "C" __global__ __aicore__ void init_kernel_bf16(
    GM_ADDR data, uint16_t num_bits, int64_t size) {
  bfloat16_t num;
  *reinterpret_cast<uint16_t*>(&num) = num_bits;
  // ...
}
```

**Host-side call**:
```cpp
half h_val = ...;
uint16_t bits = *reinterpret_cast<uint16_t*>(&h_val);
aclrtlaunch_init_kernel_fp16(..., bits, size);
```

**Trigger condition**: Any kernel with a scalar parameter that is not float/int/int64_t (especially initial-value parameters of init/fill kernels).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（P-P30，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `platform_compat`；trigger: half/bf16 scalar in extern "C" kernel params；优先级: CRITICAL。

> 索引行原文：`| P-P30 | fp16/bf16 scalar kernel arg (uint16_t bits) | platform_compat | half/bf16 scalar in extern "C" kernel params | **CRITICAL** |`
