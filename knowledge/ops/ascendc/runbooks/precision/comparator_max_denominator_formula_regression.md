---
type: "Runbook"
title: "比较器重写后 max 分母公式回退复现失败 — 公式回退当回归 + 防回退用例 + 分 dtype atol 与 worst-element 打印"
description: "GRU C++ ST 比较器迭代二已把分母修正为 max(|golden|, atol)（atol 分 dtype：fp32=1e-3、fp16/bf16=1e-2），后续 golden 量化改造重写 compare.cpp 时公式被回退成 |golden|+1e-7（注释还错标与需求文档一致），迭代三联调即复现失败——重写比较器基础设施后必须重跑曾依赖旧公式通过的边界用例；建议注释里固定规范公式原文并配防回退用例（固定数据断言 MARE 计算值）。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# 比较器重写后 max 分母公式回退复现失败 — 公式回退当回归 + 防回退用例

## 现象

GRU 算子 C++ ST 验收（CANN 9.1.0）多次迭代。迭代二已把分母从 `|golden|+1e-7` 修正为 `max(|golden|, atol)`，后续 golden 量化改造重写 `compare.cpp` 时公式被回退成 `|golden|+1e-7`（注释还错标"与需求文档公式一致"），迭代三 L1-026 复现假失败：MARE=1.80e+00（超阈）而 MERE=5.15e-06（通过），仅 13/3276800 元素不匹配（worst diff=2.967e-07/旧分母 1.647e-07 → relErr=1.80）。

## 适用场景

- **工程形态**：C++ ST 比较器多次迭代与重构的回归防护，多 dtype 算子。
- **API 路径**：`rel = |a-g| / (|g| + epsilon)` 改 `max(|golden|, atol)`，mere<th 且 mare<10×th。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.1.0。
- **适用粒度**：`operator_family`——含多轮迭代与基础设施重写的算子 ST 工程。
- **路线优先级**：`recommended_fastpath`。

## 根因

分母修复为 max 口径后，某次比较器重写时被无意回退。注释"与需求文档公式一致"加剧了回归的隐蔽性——上一轮的修复成果在代码变更中丢失，且没有回归测试捕捉。

## 解决配方

### 1. 分 dtype atol 作 max 分母 + worst-element 打印

```cpp
double atol = (dtype == DType::FLOAT32) ? 1e-3 : 1e-2;  // fp16 ulp(1.0)≈1e-3, bf16≈8e-3
double denom = std::max(std::abs(golden[i]), atol);
double rel_err = std::abs(actual[i] - golden[i]) / denom;
```

失败时打印 worst element 的 golden/actual/diff/denom——判读时一眼分辨"真错"还是"分母病"（L1-026：golden=-6.469e-08、actual=-3.614e-07、diff=2.967e-07、旧分母 1.647e-07 → relErr=1.80；规范分母 max(6.5e-8, 1e-3)=1e-3 → 2.967e-04）。

### 2. 公式回退当回归

- 重写比较器基础设施后必须重跑曾依赖旧公式通过的边界用例；
- 建议在比较器注释里固定规范公式原文；
- 配防回退用例：构造含近零 golden 的固定数据，断言 MARE 计算值。

## 验证信号

- 修复前 L1-026（fp32 大 shape，3276800 元素）：MERE=5.15e-06（通过）、MARE=1.80e+00（超阈）、13/3276800 不匹配、worst diff=2.967e-07/denom=1.647e-07；修复后 MARE=7.71e-04、L1 28/28、ST 总计 33/33；其余 27 个用例指标不劣化（来源平台：950；证据等级：tool_observed）。

## 失效边界

- atol 分档值取自生态标准在本算子的落地；其他精度等级要求（逐位对齐、混合门禁）不能直接套用。
- golden 量化改造与比较器公式是两件独立的事，不可因 golden 量化就回退公式。
- 分母修正对"数据分布导致的递归发散"无效（GRU 轨迹实测：分母已改 max + golden float 累积，MARE=9.86e+01 仍失败——权重尺度须随 fan-in 缩放才是根解，见 [RNN 权重初始化卡](rnn_st_weight_scale_fan_in_xavier.md)）。