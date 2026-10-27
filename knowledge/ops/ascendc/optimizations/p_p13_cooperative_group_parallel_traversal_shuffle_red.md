---
type: Optimization
title: "Cooperative-group parallel traversal + shuffle reduction"
description: "Transform single-thread sequential traversal into N-thread cooperative parallel + __shfl_xor divide-and-conquer reduction. cpp auto rank = threadIdx.x % GROUP_SIZE; for (uint32_t pos = rank; pos < arr"
tags: [task.optimization, topic.implementation]
status: draft
sources: []
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# Cooperative-group parallel traversal + shuffle reduction

> 原 OKF v1 卡号：**P-P13**。

## 优化点

Transform single-thread sequential traversal into N-thread cooperative parallel + `__shfl_xor` divide-and-conquer reduction.

```cpp
auto rank = threadIdx.x % GROUP_SIZE;
for (uint32_t pos = rank; pos < array_size; pos += GROUP_SIZE) { ... }
for (int32_t offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
    auto other = __shfl_xor(val, offset, GROUP_SIZE);
    if (other < val) val = other;  // or +=, max, etc.
}
```

Applicable to any scenario requiring min/max/sum within a warp.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/cooperative.md（P-P13，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `cooperative`；trigger: GROUP_SIZE parallel；优先级: HIGH。

> 索引行原文：`| P-P13 | Cooperative traversal | cooperative | GROUP_SIZE parallel | HIGH |`

## 更新 (2026-09-19)

私有沉淀仓（不公开，pinned commit cb9c78c85ead3088398f3455051355349298b3cb）域卡 `patterns/domains/cooperative.md` 复核：其内嵌 P-P13 节与本卡同 lesson（逐段比对一致），不单独建域卡；补录该域卡独有的 chip_scope 声明——warp 级协作原语（`Simt::WarpShflSync` / `Simt::WarpReduceAddSync` / GROUP_SIZE 协作遍历）仅存在于 arch35 / Ascend950PR；A2 (910B) 与 A3 (910C) 完全没有 SIMT 路径，面向这些芯片时须以 UB-scratchpad block-reduce 或 sort-then-segment-reduce 替代 warp 协作模式。
