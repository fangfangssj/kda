---
type: "Runbook"
title: "softplus_v2_grad 极值 inf 发散源于 aten 乘除顺序"
description: "softplus_v2_grad 反向在 grad≈FLT_MAX 极值用例上 golden=inf、npu=有限值(1.72e38)，根因是 aten 按 grad*exp(z)/(1+exp(z)) 顺序计算(中间 grad*exp(z) 溢出为 inf)，kernel 用安全形式 grad*(exp/(1+exp)) 不溢出，二者在极值处浮点发散。"
tags: ["task.debugging", "topic.compute.elementwise.activation", "topic.compute.elementwise.math_function", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["computation-order"]
---

# softplus_v2_grad 极值 inf 发散源于 aten 乘除顺序

## 现象

在 Ascend950PR（CANN cann-9.0.0-beta.2）上修复 SELMODE 与 L2 缓存根因后，PyTorch ST 仍有 13 个失败（后降至 10 个）。失败用例的 `grad`（gradOutput）接近 `±FLT_MAX=±3.4028e38`，`beta` 较大（如 17.0）使 `z=beta*self` 范围达 ±1700，`MERE=inf`：golden=`inf`，npu=有限值（如 `1.72e38`）。

以 L1_107 为例：`z=0.0218`（sigmoid 分支，`z≤threshold`）。golden（aten）=`inf`；npu=`1.72e38`。按 sigmoid 数学：`sigmoid(z)=1/(1+exp(-z))`，`exp(-0.0218)=0.9784`，`denom=1.9784`，`FLT_MAX/1.9784=1.72e38`（有限）。但 golden=`inf`——说明 aten 并非按 spec 的 `grad/(1+exp(-z))` 计算，而是按 `grad*exp(z)/(1+exp(z))`：`grad*exp(z)=FLT_MAX*1.022=3.48e38 > FLT_MAX → inf`，`inf/2.022=inf`。aten 在 `grad*exp(z)` 中间步骤溢出为 inf。

kernel 用安全形式 `grad * (exp(z)/(1+exp(z)))`——先算 `sigmoid=exp/(1+exp)`（不溢出），再 `grad*sigmoid`（不溢出）→ `1.72e38`，与 golden 的 inf 发散。

## 适用场景

- **工程形态**：ops-registry-invoke（算子仓开发），kernel 直调。
- **API 路径**：`AscendC::Exp`/`Adds`/`Div`/`Mul` 向量算术序列；golden = `torch.ops.aten.softplus_backward`。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN cann-9.0.0-beta.2。
- **适用粒度**：`operator_family`（所有需要对齐 aten softplus_backward 极值溢出行为的反向算子）。
- **路线优先级**：`recommended_fastpath`（按 golden=aten 的计算顺序重排 Mul/Div，先乘后除以引入中间溢出）。
- **不适用/未验证场景**：`grad` 远离 FLT_MAX 的常规输入（两种顺序等价，无发散）；非 softplus 类算子（aten 的具体顺序需各自核验 req_analysis 或 aten 源码）；a2/a3 平台未验证。

## 判别方法

**触发条件**：softplus_v2_grad（或同类 `softplus_backward`）PyTorch ST 极值用例报 FAIL，`grad` 接近 `±FLT_MAX`、`beta` 较大致 `z=beta*self` 范围大，出现 `MERE=inf`、golden=`inf` 而 npu 输出有限值（如 `1.72e38`）的发散。

**可执行探针**：用极值输入（`grad=FLT_MAX`、`z` 取 sigmoid 分支内的小正数，如 `z=0.0218`）分别按两种顺序在 fp32 下计算 `grad_sig`，比对 `torch.ops.aten.softplus_backward` 的 golden：
1. 按 aten 顺序 `(grad*exp(z))/(1+exp(z))` 计算——若 `grad*exp(z)` 越过 `FLT_MAX` → 中间为 `inf` → 结果 `inf`。
2. 按安全顺序 `grad*(exp(z)/(1+exp(z)))` 计算——`exp/(1+exp)` 不溢出 → `grad*sigmoid` 仍为有限值。
3. golden 命中哪个顺序即确认 aten 的实际计算顺序；本例 golden=`inf` 命中顺序 1（aten 先乘后除）。

**观测**：aten 实测用 `grad*exp(z)/(1+exp(z))` 顺序，`grad*exp(z)` 中间溢出为 `inf`，golden=`inf`；kernel 的安全形式输出 `1.72e38`（有限），与 golden 发散。`docs/spec.yaml` 写的 `grad/(1+exp(-z))`（安全形式）与 `docs/original_requirement/req_analysis.md §3.6.7` 记载的 aten CUDA kernel 公式 `aop * z / (z + 1)`（`z=exp(bop*beta)`，即 `grad*exp/(1+exp)` 顺序）在极值处不一致。

**判别（根因解释）**：kernel 的 `grad_sig` 用 `grad * (exp(z)/(1+exp(z)))`（先除后乘，安全），aten 用 `(grad * exp(z)) / (1+exp(z))`（先乘后除，中间 `grad*exp(z)` 可溢出为 inf）。数学等价但浮点不等价：当 `grad*exp(z) > FLT_MAX` 时 aten 溢出为 inf 而 kernel 不溢出。按 precision-debug 规则"golden 是 oracle，NPU 结果更正确不构成豁免"，kernel 须匹配 aten 的计算顺序（先乘后除），即使数值上更不安全。

## 验证信号

aten 顺序核验与诊断佐证（来源平台：950；证据等级：tool_observed）：
- PyTorch ST 极值用例 FAIL：golden=`inf`、npu=`1.72e38`、`MERE=inf`（来源平台：950；证据等级：tool_observed）。
- aten 计算顺序探针确认 aten 用 `grad*exp(z)/(1+exp(z))` 顺序（`grad*exp` 中间溢出为 inf），非 spec 的 `grad/(1+exp(-z))` 安全形式（来源平台：950；证据等级：tool_observed）。
- 按 aten 顺序重排 kernel 的 Mul/Div（配合 DFX_IN 修复）后，L1_107/L1_094/L1_250 等极值用例由 FAIL 转 PASS，批跑 568 pass / 10 fail（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅当 `grad*exp(z)` 跨越 `FLT_MAX` 时发散；常规输入（`grad` 远离 `FLT_MAX`）两种顺序等价。
- aten 其他算子的具体计算顺序须各自核验 req_analysis 或 aten 源码，不可直接套用本卡结论。
- a2/a3 平台未验证，推断同 fp32 溢出模型但未实测。