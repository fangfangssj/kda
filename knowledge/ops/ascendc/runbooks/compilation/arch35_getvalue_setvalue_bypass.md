---
type: "Runbook"
title: "arch35 GetValue/SetValue 标量bypass禁用"
description: "在 arch35 RegBase 向量 kernel 中对标量输入（count==1）用 LocalTensor::GetValue/SetValue 做标量计算 bypass Compare+Select，Python ST 失败数远劣于 baseline（回退后恢复），scalar/vector API 混用有严重副作用，banned。"
tags: ["task.debugging", "topic.programming_model.execution_model.simd", "topic.compute.elementwise.comparison_logic"]
status: stable
sources: []
created_at: "2026-08-18T05:55:18Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["scalar_bypass"]
---

# arch35 GetValue/SetValue 标量bypass禁用

## 现象

在 Ascend950PR（arch35 / RegBase、CANN cann-9.0.0-beta.2）上开发 `softplus_v2_grad` 算子。为绕过 `Select(VSEL_CMPMASK_SPR)` 在标量场景（count==1）的 mask 污染问题，尝试对标量输入用 `LocalTensor::GetValue` / `SetValue` 做标量计算 bypass：当 count==1 时用 GetValue 从 LocalTensor 取标量值、用标量数学函数计算、用 SetValue 写回，绕开 Compare+Select 路径。

结果：Python ST 失败数暴涨——远劣于 baseline（53 失败），也远劣于改用 `VSEL_TENSOR_TENSOR_MODE` 的对照（52 失败），整个 kernel 行为被破坏。回退该 bypass 后失败数恢复到 baseline 水平。

## 适用场景

- **工程形态**：kernel 直调（ops-registry-invoke，RegBase，arch35 / Ascend950PR）。
- **API 路径**：`LocalTensor<T>::GetValue` / `LocalTensor<T>::SetValue`（在 `op_kernel/arch35/softplus_v2_grad_kernel.h` 的 `ComputeFp32Core` 中）。
- **验证过的平台与 CANN 版本**：Ascend950PR（注册平台 `950`），CANN cann-9.0.0-beta.2，arch 目录 arch35，Python ST 581 例实测。
- **适用粒度**：`platform_wide`——arch35 / Ascend950PR 上所有向量 kernel 中试图用 GetValue/SetValue 做 scalar bypass 的场景。
- **路线优先级**：**banned**——GetValue/SetValue scalar bypass 在 arch35 向量 kernel 中有严重副作用，不得采用；即使输入是标量（count==1）也应使用向量 API（Muls/Exp/Div/Select）处理，用 count_padded 对齐到 vector lane size。
- **明确不适用/未验证场景**：a2/a3 平台未验证；非 arch35 芯片上 GetValue/SetValue 行为未验证；count >= vector lane size 的大 shape 场景不涉及此 bypass（不需要标量路径）。

## 失败路线

**banned 路线：在 arch35 RegBase 向量 kernel 中对 count==1 用 `LocalTensor::GetValue` / `SetValue` + 标量数学函数做 bypass，绕开 Compare+Select。**

- **尝试**：在 `ComputeFp32Core` 内加 `if (count == 1)` 标量分支，用 GetValue 取标量、标量数学函数计算、SetValue 写回。先后试过用标量 `expf()` 与向量 `Exp` + GetValue/SetValue 组合两种写法。
- **客观失败**：标量 bypass 后 Python ST 失败数远劣于 baseline，远劣于 `VSEL_TENSOR_TENSOR_MODE` 对照（来源平台：950；证据等级：tool_observed）。
- **回退**：移除该 scalar bypass 后失败数恢复到 baseline 水平（来源平台：950；证据等级：tool_observed）。
- **边界**：本配置（arch35 / Ascend950PR、count==1 标量分支、RegBase 向量 kernel）下该 bypass 路线 banned；机制层根因未追明（见 `## 根因`），不得以此卡暗示"已找到修复"。

## 根因

**未追到机制层根因。** arch35 RegBase 向量 kernel 中 `LocalTensor::GetValue` / `SetValue` 是标量访问接口，在向量计算流中混用 scalar API 与 vector API（Muls/Exp/Div/Compare/Select）会破坏向量计算管线的一致性。确切机制未诊断——可能涉及：

1. GetValue/SetValue 与向量 DMA 操作的同步问题（标量访问可能在向量操作未完成时读取/写入）。
2. arch35 上 LocalTensor 的标量访问路径与向量访问路径的寄存器/buffer 冲突。
3. SetValue 写回的标量值在后续向量操作中被覆盖或未正确对齐。

> 注：确切机制未诊断。观察到失败数暴涨后立即回退，未深入分析。bypass 代码未从 trace 事件流完整恢复——edit 事件（seq 356/363）的 content 被 markdown_transcript adapter redacted（显示 '_see below_'），完整代码无法从事件流恢复；上述结构仅从 reasoning 事件（seq 362-363）推断，非 verbatim。

## 验证信号

本段只收本 trace 内 tool_observed 实测，agent 推断的机制假设不进此段。

- 标量 GetValue/SetValue bypass 启用后 Python ST 失败数远劣于 baseline（53 失败），亦远劣于 `VSEL_TENSOR_TENSOR_MODE` 对照（52 失败）（来源平台：950；证据等级：tool_observed）
- 移除该 scalar bypass 后失败数恢复到 baseline 水平（来源平台：950；证据等级：tool_observed）

## 坏实践

- 在 arch35 向量 kernel 中对 count==1 用 GetValue/SetValue + 标量数学函数做 bypass——scalar/vector API 混用破坏向量计算管线，失败数暴涨。
- 先用 `expf()`（标量 C math）尝试——可能不适用于 arch35 kernel 编译环境；后改用向量 `Exp` + GetValue/SetValue 组合，仍失败。

## 预防

- 在 arch35 RegBase 向量 kernel 中，即使输入是标量（count==1），也应使用向量 API（Muls/Exp/Div/Select）处理，用 count_padded 对齐到 vector lane size；不得用 GetValue/SetValue 做 scalar bypass。
- 如需对标量输入做条件选择，优先用 `Select(VSEL_TENSOR_TENSOR_MODE)` 或纯算术方式（Sub/Muls/Mins/Maxs/Mul/Add），不混用标量 API。

## 失效边界

- 平台：本 trace 只在 Ascend950PR（`950`）实测。a2/a3 上 GetValue/SetValue 在向量 kernel 中的副作用预期类似，但**未验证，推断可迁移**；如要落到 a2/a3 须重测。
- 确切失败机制未追到硬件层——失败是 compile-time 还是 runtime 错误未区分（从 trace 看编译通过、运行时产出错误值）。
- 本卡结论限定于"标量 bypass 在 arch35 向量 kernel 中混用 scalar/vector API"这一路线；GetValue/SetValue 在合法标量上下文（如 host 侧、非向量流）的正当用途不受影响。