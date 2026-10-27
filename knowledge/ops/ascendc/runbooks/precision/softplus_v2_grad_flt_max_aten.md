---
type: "Runbook"
title: "softplus_v2_grad 极端 FLT_MAX 与 aten 不可对齐"
description: "softplus_v2_grad 在 gradOutput=±FLT_MAX 或 threshold=858 超出 spec extreme_inputs 的极端应力输入下，NPU 稳定式 gradOutput×sigmoid(z) 与 aten 不稳定式 aop×exp(z)/(exp(z)+1) 因计算顺序与硬件 FTZ 在 fp32 表示极限处不可对齐；软件无法修复，banned。"
tags: ["task.debugging", "topic.compute.elementwise.activation", "topic.compute.elementwise.math_function", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-18T15:44:46Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["intermediate_overflow", "stable_form"]
---

# softplus_v2_grad 极端 FLT_MAX 与 aten 不可对齐

## 现象

在 Ascend950PR 上为 `softplus_v2_grad` 修复 3 个 bug 后，PyTorch ST 全量仍残留 7 个失败用例。这些用例的输入属性全部为 `gradOutput=±FLT_MAX(3.4028235e+38)` 或 `threshold=858`，超出 `spec.yaml` `extreme_inputs` 定义范围（spec 仅定义 NaN/+inf self、大负数 self、fp16 上溢边界，默认 threshold=20）。残留 case 的特征：

- `gradOutput=+FLT_MAX, z>0`：golden(aten)=+inf，NPU=有限值（`gradOutput×sigmoid(z)`，sigmoid 不溢出）；
- `gradOutput=-FLT_MAX, z>0`：golden=-inf，NPU=有限负值；
- `threshold=858, self=±65504(fp16)`：`exp(858)` 溢出 inf，`inf/inf=NaN`，golden 与 NPU 均产 NaN，仅边界 z 有单侧 NaN 差异。

## 适用场景

- 工程形态：kernel 直调 + aclnn（golden 为 aten CPU fp32 参考）。
- API 路径：`torch.ops.aten.softplus_backward`（golden）vs NPU `softplus_v2_grad` Kernel。
- 验证平台与版本：Ascend950PR，CANN cann-9.0.0-beta.2；golden 为 CPU fp32 aten。
- 适用粒度：`operator_family`——softplus 家族及任何 `gradOutput×sigmoid(z)` 稳定式与 aten `aop×exp(z)/(exp(z)+1)` 不稳定式对照的算子。
- 路线优先级：`banned`——禁止为对齐这些极端值而改动 NPU 稳定式或放宽容差。
- 不适用/未验证场景：spec `extreme_inputs` 范围内的边界（NaN/+inf self 等）不属本卡；这些用例应通过 spec 内的精度策略处理。

## 根因

残留失败是 fp32 表示极限 + golden(aten) 自身中间溢出 + 硬件 FTZ 的共同结果，属 golden/硬件限制而非算子算法 bug。aten 按 `aop × exp(z)` 先算，对 `gradOutput=FLT_MAX, z>0`，`aop × exp(z)` 先溢出为 ±inf，再除 `(exp(z)+1)` 得 ±inf；NPU 按 spec 推荐的稳定式 `gradOutput × (exp(z)/(1+exp(z)))`（先算 sigmoid 再乘）不溢出，得有限值（数学上更正确）。二者因计算顺序不同在 fp32 表示极限处不可对齐。对 `z∈[-95,-38]`，`exp(z)` 为 fp32 非规范数，NPU V pipe FTZ（Flush-To-Zero）刷为 0，golden（CPU 保留非规范数）得小非零值，`gradOutput=FLT_MAX` 放大差异。上述 aten 运算顺序与 FTZ 行为的归因来自开发会话的离线分析重算，非本卡上板实测信号。

## 失败路线

**banned：不得为对齐 `gradOutput=±FLT_MAX`/`threshold=858` 这 7 个 case 而改动 NPU 稳定式或放宽容差。**

尝试与回退：为对齐这些极端值，曾按 golden(aten) 的运算顺序 `aop × exp(z) / (exp(z)+1)` 重算，验证是否可在 NPU 侧复现 aten 的中间溢出以对齐。结果：NPU 稳定式数学上更正确（不溢出），按 golden 不稳定式重算才能复现其中间溢出——这意味着对齐只能通过引入数学上更差、且 spec 明确不推荐的运算顺序达成，得不偿失。三个候选修复均被放弃：

- 候选 A：把 NPU 稳定式改成 aten 不稳定式以复现中间溢出——放弃：破坏正常 case 的数值稳定性，且 spec 明确推荐稳定式。
- 候选 B：放宽容差覆盖这些 case——banned（放宽容差不是修复）。
- 候选 C：在 Kernel 侧检测 FLT_MAX 输入并 special-case 复现 aten 溢出——放弃：为对齐 golden 的"错误"行为而引入硬件不自然的分支。

边界：`gradOutput=±FLT_MAX` 或 `threshold=858` 且超出 spec `extreme_inputs` 范围时，NPU 稳定式与 aten 不稳定式不可对齐——软件无法修复。

## 设计决策

不采用任何"对齐"方案。正确做法：把这些 case 定性为 golden/硬件限制、超出 spec 范围，豁免结论交用户确认。spec `extreme_inputs` 范围内的边界仍须正确处理，不在此豁免之列。

## 预防

- 精度回归前先确认 spec `extreme_inputs` 范围，超出范围的应力用例不作为算子正确性判据。
- 对 `gradOutput=±FLT_MAX` 类输入，提前识别 golden(aten) 自身中间溢出风险，避免盲目追修。
- 对照 golden 与 NPU 运算顺序时，先判断 golden 行为本身是否数值正确，再决定是否对齐。

## 验证信号

以下信号来自本次开发会话在 Ascend950PR 上的 PyTorch ST / C++ ST 全量回归。

- 修复 3 个 bug 后残留 7 个失败用例，其输入属性均为 `gradOutput=±FLT_MAX(3.4028235e+38)` 或 `threshold=858`，超出 spec `extreme_inputs` 范围（来源平台：950；证据等级：tool_observed）。
- 同次回归 L0 51/51 PASS、C++ ST 57/57 PASS、全量 571/578（残留即本 7 例）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本结论仅对 `gradOutput=±FLT_MAX`/`threshold=858` 且超出 spec `extreme_inputs` 的输入成立；spec 范围内的极端值（NaN/+inf self、fp16 上溢边界）仍需正确处理。
- a2/a3 是否有相同 V pipe FTZ 与 fp32 表示极限未验证，不外推。