---
type: Optimization
title: "Cooperative-group value transport"
description: "GROUP_SIZE threads share the work of a large vector copy: cpp for (uint32_t j = rank; j < dim; j += GROUP_SIZE) { dst[pos dim + j] = src[idx dim + j]; } At dim=128: 128 stores → 16 threads each do 8 s"
tags: [task.optimization, topic.programming_model.execution_unit.vector]
status: draft
sources: []
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# Cooperative-group value transport

> 原 OKF v1 卡号：**P-P16**。

## 优化点

GROUP_SIZE threads share the work of a large vector copy:
```cpp
for (uint32_t j = rank; j < dim; j += GROUP_SIZE) {
    dst[pos * dim + j] = src[idx * dim + j];
}
```

At dim=128: 128 stores → 16 threads each do 8 stores. Applicable to any large embedding/feature vector transport.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/cooperative.md（P-P16，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `cooperative`；trigger: large vector copy；优先级: MEDIUM。

> 索引行原文：`| P-P16 | Cooperative value copy | cooperative | large vector copy | MEDIUM |`

## 更新 (2026-09-19)

私有沉淀仓（不公开，pinned commit cb9c78c85ead3088398f3455051355349298b3cb）域卡 `patterns/domains/cooperative.md` 复核：其内嵌 P-P16 节与本卡同 lesson（逐段比对一致），不单独建域卡；补录该域卡独有的 chip_scope 声明——warp 级协作原语（`Simt::WarpShflSync` / `Simt::WarpReduceAddSync` / GROUP_SIZE 协作遍历）仅存在于 arch35 / Ascend950PR；A2 (910B) 与 A3 (910C) 完全没有 SIMT 路径，面向这些芯片时须以 UB-scratchpad block-reduce 或 sort-then-segment-reduce 替代 warp 协作模式。
