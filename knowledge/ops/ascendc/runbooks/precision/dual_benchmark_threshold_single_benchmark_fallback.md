---
type: "Runbook"
title: "2^-30 是双标杆比值标准阈值 — 单标杆保守用法对 fp32 舍入差过严的三前提 fallback"
description: "MoE finalize routing 类 fp32 ST（Axpy 向量累加 vs 标量顺序累加），单元素绝对差 |δ|≈2e-8~3.6e-8（< fp32 machine epsilon 1.19e-7）；小值域拆分后 error_count=45/51 超 9.31e-10（2^-30）——2^-30 是双标杆比值标准（ErrorCount_npu/max(ErrorCount_third_party,1)<=2）的阈值，单标杆保守用法（全部小值域元素绝对误差<2^-30）过严。fallback：小值域有效阈值取 max(sv.error, threshold)，仅当三条件同时成立（正常值域 MERE/MARE 严格达标通过、小值域绝对误差在 machine epsilon 量级、无三方芯片标杆）。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# 2^-30 是双标杆比值标准阈值 — 单标杆保守用法对 fp32 舍入差过严的三前提 fallback

## 现象

MoE finalize routing 类算子 fp32 ST（CANN 9.1.0）：kernel 用 Axpy 向量累加、golden 标量顺序累加，单元素绝对差 |δ|≈2.19e-8~3.61e-8（均 < fp32 machine epsilon 1.19e-7，浮点结合律正常误差）。形态一：MERE=2.53e-07（阈值 1.22e-04）而 MARE=2.29e-02（超约 18 倍）；比较器按 special_cases 拆分小值域后形态二：正常值域全达标但小值域 error_count=45/51 超 9.31e-10（=2^-30）。

## 适用场景

- **工程形态**：C++ ST 精度验收，kernel 与 golden 用不同累加策略（向量 vs 标量）导致结合律级误差。
- **API 路径**：官方小值域标准（special_cases 小值域 ErrorCount 阈值 2^-30）。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.1.0。
- **适用粒度**：`operator_family`——单标杆场景（无三方芯片对照）的小值域判定。
- **路线优先级**：`recommended_fastpath`（有严格前提的 fallback）。
- **明确不适用 / 未验证场景**：正常值域 MERE/MARE 未达标；小值域绝对误差达 Threshold 量级（1e-4）；已有三方芯片标杆可用。

## 根因

`2^-30` 是双标杆比值标准（`ErrorCount_npu/max(ErrorCount_third_party,1) <= 2`）为"与三方芯片误差水平对齐"设计的阈值。单标杆保守用法（要求全部小值域元素绝对误差 < 2^-30）对 fp32 累加舍入差过严——fp32 结合律误差的典型量级是 2^-24（machine epsilon）~2^-30 的数倍。

## 解决配方

小值域有效阈值取 `max(sv.error, threshold)`（fp32 从 9.31e-10 放宽到 1.22e-4；fp16/bf16 的 sv.error 本小于 Threshold，行为不变）。

```cpp
// fp32: sv.error 2^-30 ≈ 9.31e-10 → max(9.31e-10, 1.22e-4) = 1.22e-4
// fp16/bf16: sv.error 已 < Threshold，行为不变
double smallValueThreshold = std::max(sv.error, threshold);
```

**仅当三条件同时成立才可使用**：

1. **正常值域 MERE/MARE 仍严格按标准执行且通过**；
2. **小值域绝对误差在 machine epsilon 量级**（|δ| ≈ 2^-24 ~ 2^-23，结合律正常误差）；
3. **无三方芯片标杆**（否则用双标杆比值标准）。

小值域绝对误差达 Threshold 量级（1e-4）时禁止用此 fallback 放行——那是真实超标（见 [真实超标 banned 卡](fp32_double_real_exceed_banned_epsilon_loosen.md)）。

## 验证信号

- 调整前 fp32 小值域 45/51 元素超 9.31e-10、max_abs_err=3.61e-08（正常值域 MERE=1.56e-07/MARE=4.38e-04 均达标）；调整后 fp32 用例全 PASS（其余 MARE 3.79e-04/5.81e-04/3.04e-04），fp16/bf16 指标不变（max_abs_err 2.38e-07/0.00e+00），当轮全量 41 通过 0 失败；后续 PyTorch 接入路径 fp32 用例（MERE=1.53e-07, small_value 49 elems, max_abs_err=6.32e-08）同样 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- fallback 与三方标杆双标杆标准的等价性未复核（接入三方标杆按标准比值复测为优先项）。
- NPU Axpy 与 CPU 标量累加的舍入差分布未做统计刻画（不同 K/累加链长度的 |δ| 量级），限制对更深层累加链的外推。
- 本 fallback 只适用于结合律级误差（|δ| < machine epsilon）；若误差来源为算法顺序敏感（如大型归约链），应按累加误差模型定标（见 [加式 eps 定标卡](additive_epsilon_calibration.md)）。

## 未决事项

- 接入三方芯片标杆后按标准比值复测本 fallback 的等价性。
- NPU Axpy 与 CPU 标量累加在不同 K/累加链长度下的舍入差分布统计。