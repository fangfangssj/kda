---
type: Optimization
title: "ReduceMin/ReduceMax calc_index=true 实现向量化 argmin/argmax"
description: "用 ReduceMin/ReduceMax(calc_index=true) 并重解释编码索引位，得到向量化的 argmin/argmax，避免标量收尾循环。原 legacy 锚 P-P252（冲突 ID P-P62）。"
tags: [task.optimization, topic.compute.reduction, topic.programming_model.execution_unit.vector]
status: draft
sources: []
created_at: "2026-09-22T05:30:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# ReduceMin/ReduceMax(calc_index=true) 做向量化 argmin/argmax

需要 argmin/argmax 时，用 `ReduceMin`/`ReduceMax` 的 `calc_index=true` 形式并重新解释返回的编码索引位，而不是手写标量/循环收尾。这是硬件自带的索引追踪支持。

适用：归约类核里需要最值位置的场景。注意编码位的解释方式以对应 API 的具体字段为准（参数字段的语义逐 API 核对，别从邻近 API 类推——同类教训见 repeat stride 的 element vs block 粒度差异）。

相关：[半区间树归约](p_p47_half_interval_tree_reduction_in_place.md)（D>64 时的归约折叠；其文末也指向本模式）。
