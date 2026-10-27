---
type: Runbook
title: "precision_eval_two_tier 控制台行的 legacy MERE/MARE 列恒显示 0.00e+00——真指标在 JSON results[].outputs[]，别把控制台 0 当成退化比较或逐位一致"
description: "precision_eval_two_tier 控制台行的 legacy MERE/MARE 列恒显示 0.00e+00——真指标在 JSON results[].outputs[]，别把控制台 0 当成退化比较或逐位一致"
tags: [task.testing, topic.precision]
status: draft
sources: []
created_at: "2026-09-20T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-EVAL-LEGACY-MERE-COLUMN-ZERO"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-EVAL-LEGACY-MERE-COLUMN-ZERO

## 适用场景

- 所有用 precision_eval_two_tier.py 的精度评估（全平台全 CANN）。

## 结论（39_ChunkKdaFwd 实测）

冻结 profile 下控制台打印的 `ours_M=0.00e+00/0.00e+00` 是 legacy MERE/MARE 列未填充的假象；真实指标在 JSON `results[].outputs[]`。读控制台的两个 0 既不代表「比较退化」也不代表「逐位一致」。

- **Evidence**：39 归档（verified_on: Ascend950DT_9582, CANN 9.2.0）。
- **Cross-ref**：CAND-PRECISION-ALLCLOSE-VACUOUS-SMALL-OUTPUT。
