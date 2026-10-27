---
type: Runbook
title: "De-scalarize flash online-softmax——手写 mem-based VEC online-softmax + 精度三件套（替换 SoftmaxFlashV2 标量极）"
description: "De-scalarize flash online-softmax——手写 mem-based VEC online-softmax + 精度三件套（替换 SoftmaxFlashV2 标量极）"
tags: [task.optimization, topic.precision]
status: draft
sources: []
created_at: "2026-09-21T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["P-P101"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# P-P101 — De-scalarize flash online-softmax——手写 mem-based VEC online-softmax + 精度三件套

## 适用场景

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=attention-softmax（FA 前后向、masked-LM、任何 flash online-softmax）` `unverified_on: Ascend910_V220（技术原则上架构无关，A3 未实测）`

## 正文（legacy PATTERN_INDEX 原文转卡，实测证据在内）

Attention/FA 的 softmax 被标量绑死时（aiv_scalar 主导管道：A5 FA 0.27–0.31 耗在 SoftmaxFlashV2 内，A3 FA 0.503），用手写 mem-based VEC online-softmax 替换 `SoftmaxFlashV2`/标量 `RowMuls`/`RowDivs`：标准 LocalTensor 向量算子（WholeReduceMax / 减行最大值后 Exp / WholeReduceSum / Muls / Div）替换标量极；**精度三件套**（masked→有限 minValue 而非 −inf / 先减行最大值再 Exp / 向量 Div 且 sum>0）修掉 inf bug；**陷阱**：stat-buffer 别名（V 管 Brcb 与 MTE3 sm-emit 别名 → sm_sum 损坏 → inf）。

**性能作用域（kernel≠wall）**：kernel-msprof −24% 但 e2e-wall 中性（host-fold+pybind 主导，见 OL-201）——先测 wall 分解，别假设 kernel 收益传导到 e2e。架构原则上无关（V220+V351）。完整机制见 legacy `fa_class/cv_reference_concrete_params.md`。

- **Evidence**：A5/CANN-9.0.0 精度 5/5（commit 5a3a1cee v2 / 697ed8c8 v5，FA-A5 path-B 2026-05-31）。
- **Cross-ref**：OL-201（host-fold 主导）、P-P102/P-P116（MIX 脚手架）。

`verified_on: soc=Ascend950PR; cann=9.0.0 b226; bisheng=15.0.5（FA-A5 path-B 2026-05-31，5/5 精度）`
