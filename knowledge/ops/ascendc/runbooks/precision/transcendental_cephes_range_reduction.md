---
type: Runbook
title: "手写超越函数走 Cephes 式 range reduction + fp32 级原语——端到端撞精度天花板且 vendor 源确认算法改写时"
description: "手写超越函数走 Cephes 式 range reduction + fp32 级原语——端到端撞精度天花板且 vendor 源确认算法改写时"
tags: [task.implementation, topic.precision]
status: draft
sources: []
created_at: "2026-09-21T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["P-P88"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# P-P88 — 手写超越函数 Cephes 式 range reduction（fp32 级原语）

## 适用场景

`applies_to: soc=Ascend950PR（A5 实测）+ Ascend910_V220（A3 交叉确认）; cann=9.0.0+; op_class=含 Tanh/Sigmoid/GELU/SwiSH 等超越函数的算子`

## 正文（legacy PATTERN_INDEX 原文转卡，实测证据在内）

端到端超越函数 kernel 撞精度天花板且 vendor 源确认算法改写时的出路。**三条独立证据流交叉验证（2026-05-07）**：

1. **PB-24/25 隔离原语实测（A5/Ascend950PR）**：`Tanh<fp32>` 双峰地板——|x|≥0.1 干净 2 ULP，小 x 灾难性失败（x≈1.7e-4 时 1599 ULP，x=1e-7 时 2.7M ULP），多项式在小 x 不保 tanh(x)≈x。`Sigmoid<fp32>` 是干净的均匀 2-ULP。
2. **A3（910 V220）跨架构确认**：同样的双峰 Tanh 地板——失败模式是**芯片家族级**的，不是架构特定。
3. **vendor 源证据**：CANN 自家 arch35 GELU 根本不调 `Tanh` 原语——用 `MicroAPI::Mul/Axpy/Muls/Exp/Adds/Div` 实现 sigmoid 形式改写（fp16/bf16 先升 fp32 再 RINT 回）。

**Canonical Cephes 形式**：大 |y| 时 `tanh(y) = 1 − 2/(exp(2y)+1)`（消掉饱和带 (1+tanh) 抵消）；`sigmoid(y) = 1/(1+exp(−y))` 直接 Exp+Reciprocal+Add（**不用** AscendC Sigmoid——fp16 级）。Cephes/fdlibm/libm/ARM CL 文档化的 range-reduction 算法只用 fp32 级原语就能到 sub-ULP。

**工作流**：① 用 API 替换探针定位精度瓶颈原语（如 GELU iter1 Tanh→Sigmoid 恒等替换回归 → 确诊原语天花板）；② 查公开算法库文档；③ 验证所用原语在平台上的实测地板。**诚实边界**：把该模式推广到其他超越函数（Erf/Atan 等）仍需隔离原语探针证据；对 GELU 而言 vendor 源证据已足够直接引用。

- **Evidence**：PB-24/25（A5 隔离实测）+ A3 DS 侧交叉确认 2026-05-07 + CANN ops-nn gelu_dag.h vendor 源。
- **Cross-ref**：PB-24/PB-25（原语地板）、OL-103（Tier-2 超越函数精度方法论）。

`verified_on: 三证据流 2026-05-07（A5 PB-24/25 + A3 DS + vendor 源）`
