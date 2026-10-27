---
type: Optimization
title: "Radix Sort Twiddle Transform for Float/Signed Types"
description: "Problem: Radix sort sorts by bit pattern. IEEE 754 float bit patterns do not directly reflect numeric magnitude (negative bit patterns are \"larger\" than positive ones). Twiddle transformation rules: -"
tags: [task.optimization, topic.programming_model.execution_unit.vector, topic.implementation.masking]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p44-radix-sort-twiddle-transform-for-float-signed-ty.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Radix Sort Twiddle Transform for Float/Signed Types

> 原 OKF v1 卡号：**P-P44**。

## 优化点

**Problem**: Radix sort sorts by bit pattern. IEEE 754 float bit patterns do not directly reflect numeric magnitude (negative bit patterns are "larger" than positive ones).

**Twiddle transformation rules**:
- **Unsigned int**: no transform needed
- **Signed int**: XOR sign bit mask (`0x80000000` for int32)
- **Float**: positive XOR `0x80000000` (flip sign bit); negative XOR `0xFFFFFFFF` (flip all bits)
- **Descending**: additional NOT of all bits after the transform

```cpp
// Vectorized implementation:
And(signBits, data, 0x80000000);         // extract sign bit
CompareNE(mask, signBits, 0);             // mask for negatives
Select(xorMask, mask, 0xFFFFFFFF, 0x80000000);  // pick XOR mask
Xor(twiddled, data, xorMask);            // apply transform
```

**Evidence**: CANN sort_radix_sort_more_core.h:314-562. E1 level.

**Stop condition**: Only needed for radix sort. The hardware Sort API (Concat+Sort+Extract) handles this internally.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/sort.md（P-P44，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `sort`；trigger: bit-twiddle transform for float/signed radix sort；优先级: MEDIUM。

> 索引行原文：`| P-P44 | Radix sort twiddle transform | sort | bit-twiddle transform for float/signed radix sort | MEDIUM |`
