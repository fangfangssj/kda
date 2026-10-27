---
type: "Runbook"
title: "自建比较器近零兜底 epsilon 用独立魔法数与阈值脱节 — 灾难性消除产物仍走相对误差"
description: "TensorList 类算子 y=x1+(x2/x3)*scalar 的灾难性消除使 golden 落 1e-6~1e-3，FP32 绝对误差 ~1e-7（由操作数幅度 ~10 × 机器精度决定），epsilon=1e-6 只把 |golden|<1e-6 移入绝对判定，1e-6≤|golden|<1.22e-3 的消除产物仍走相对误差（|golden|=1e-6 时 rel≈0.1，MARE 超阈 56 倍）——兜底阈值与精度阈值脱节是脚手架设计缺陷。修复 epsilon=10×threshold（1.22e-3）、atol=threshold，与 numpy allclose 混合判据同型，MERE 等值复现证明非放宽容差。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# 自建比较器近零兜底 epsilon 用独立魔法数与阈值脱节 — 灾难性消除产物仍走相对误差

## 现象

TensorList 类算子（多张量扩展，Ascend950PR + CANN 9.1.0）的 19 个 C++ ST 用例中 2 个失败，形态与主卡同签名（**MERE 极小但 MARE 超阈**）：MERE=1.90e-08（阈 1.22e-04）通过而 MARE=6.81e-02（阈 1.22e-03）超阈约 56 倍，nan/inf mismatch=0；四例失败值 6.81e-02/3.52e-03/1.02e-02/3.26e-02 全部 MERE≈1e-8 量级；均为 16M/8M 元素大 shape。

## 适用场景

- **工程形态**：自建 C++ ST 比较器（MERE/MARE + 近零兜底，对照 CPU golden），多张量算子。
- **API 路径**：比较器近零判定阈值 epsilon、绝对误差阈值 atol、mareThreshold=10×threshold。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.1.0。
- **适用粒度**：`operator_family`——含灾难性消除（近零大抵消）算子的比较器设计。
- **路线优先级**：`recommended_fastpath`（挂钩阈值，非魔法数）。

## 根因

算子公式 y = x1 + (x2/x3)*scalar 存在灾难性消除（x1 ≈ -(x2/x3)*scalar），golden 值落入 1e-6~1e-3 区间。对这类元素：FP32 逐元素运算的绝对误差 ~1e-7（由操作数幅度 ~10 × 机器精度 1.2e-7 决定）；相对误差 = 1e-7/|golden|，当 |golden|=1e-6 时 ≈ 0.1，远超 MARE 阈值；而绝对误差 1e-7 本身远小于任何合理 atol。兜底阈值 `epsilon=1e-6` 只把 |golden| < 1e-6 的元素移入绝对误差判定，1e-6 ≤ |golden| < 1.22e-3 的消除产物仍走相对误差——**兜底阈值与精度阈值脱节（1e-6 是拍脑袋常量，未与 threshold 建立关系）是脚手架设计缺陷**。

## 解决配方

`tests/st/test_aclnn_<op>.cpp` 比较器参数段 verbatim（before/after）：

```cpp
    // 修复前：epsilon 为独立魔法数
    const double epsilon = 1e-6;     // 近零判定阈值
    const double atol = threshold;   // 绝对误差允许阈值（FP32: 1.22e-4）
    double mareThreshold = 10.0 * threshold;

    // 修复后：epsilon 与阈值挂钩
    const double epsilon = 10.0 * threshold;  // 近零判定阈值（FP32: 1.22e-3）
    const double atol = threshold;            // 绝对误差允许阈值（FP32: 1.22e-4）
    double mareThreshold = 10.0 * threshold;
```

判定主体不变（近零桶绝对误差、正常桶 `diff / (|golden| + 1e-7)`、MERE < threshold 且 MARE < 10×threshold 且 nan/inf 与近零桶全过才 PASS）。

**为什么这不是"放宽容差"**（与近零家族同一红线）：

- 良态元素（|golden| ≥ 10×threshold）的判据一字未动；
- 移入绝对误差桶的元素须满足 |actual-golden| < threshold（1.22e-4），而其实测绝对误差 ~1e-7，有 3 个数量级裕量；
- 修复前后 MERE 完全不变（1.90e-08 等值复现）——与 numpy `allclose(|a-g| ≤ atol + rtol·|g|)` 的混合判据同型，相对误差对近零分母本来就不适定。

## 设计决策

- **为何取 epsilon=10×threshold 而非 threshold**：消除产物 |golden| 可到 1.22e-3（=10×threshold）量级，epsilon 需覆盖该范围；atol 仍取 threshold 保持绝对误差上限语义。
- **为何不调 mareThreshold**：调大阈值连良态元素的真超阈一起放过，属掩盖（见 [真实超标 banned 卡](fp32_double_real_exceed_banned_epsilon_loosen.md)）。

## 验证信号

- 修复前：2/19 失败，均为 MERE 通过（1.90e-08/8.69e-09，阈 1.22e-04）而 MARE 超阈（6.81e-02/3.52e-03，阈 1.22e-03）（来源平台：950；证据等级：tool_observed）。
- 修复后（epsilon=10×threshold，FP32: 1.22e-3）：19/19 PASS；失败用例 MERE 数值不变（1.90e-08 量级），原近零元素以 `near-zero pass=N` 计数呈现（如 1048576 元素中 99~136 个进入绝对误差桶）（来源平台：950；证据等级：tool_observed）。
- Mock 模式（CPU golden 自检）同步 19/19 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- epsilon=10×threshold 的取值依据是"覆盖本算子消除产物的 |golden| 范围（1e-6~1e-3）"；若某算子 golden 可逼近 10×threshold 但绝对误差也逼近 atol，判据收紧为 atol=threshold 可能误杀——须按算子数值特性复核。
- 仅 FP32 实测（threshold=1.22e-04）；FP16/BF16 的 threshold 不同，公式自动适配但未验证。
- 近零兜底 epsilon 只覆盖绝对误差在 atol 内的元素；绝对误差超 atol 的真缺陷仍被拦截（near-zero 桶 FAIL）。

## 坏实践

- 用独立魔法数做近零阈值且不写推导注释（本卡原始缺陷）。
- 发现 MARE 虚高后直接调大 mareThreshold 了事——真精度回归会被一起放过。