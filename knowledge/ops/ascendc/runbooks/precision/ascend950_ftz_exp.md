---
type: "Runbook"
title: "Ascend950 向量管 FTZ 刷新次正规 exp 为零"
description: "Ascend950(dav_c310) Exp 向量指令对落入 fp32 次正规区间的输出走 FTZ 刷新为 0，CPU golden 保留次正规；grad≈FLT_MAX 时残差被放大到正常数域，3 用例稳定发散，kernel 侧不可修，banned。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.math_function"]
status: stable
sources: []
created_at: "2026-08-18T08:28:22Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Ascend950 向量管 FTZ 刷新次正规 exp 为零

## 现象

在 Ascend950（arch35 / dav_c310）、CANN 9.0.0-beta.2 上，`softplus_v2_grad` kernel 修复全部四类可修软件根因（Select SELMODE 误用、L2 DFX_IN 缓存键缺标量、计算顺序与 aten 不一致、`-50` 下界 clamp 阻止下溢）后，PyTorch ST 仍有 3 个稳定失败：`L1_018`、`L1_083`、`L1_281`。三例均满足 `grad=±FLT_MAX` 且 `z=beta*self` 落在 `[-104, -87]` 区间。该区间 `exp(z)` 为 fp32 次正规数（如 `exp(-87)≈1.76e-40 < FLT_MIN=1.175e-38`）。CPU golden（aten）保留次正规结果，`golden = FLT_MAX × 1.76e-40 ≈ 6e-2`（正常值）；NPU 向量管 `Exp` 指令对次正规输出执行 FTZ（Flush-To-Zero）刷新为 0，`npu = FLT_MAX × 0 = 0`。发散可见且不可由上述四类软件根因解释。

## 适用场景

- **工程形态**：ops-registry-invoke（算子仓开发），kernel 直调。
- **API 路径**：`AscendC::Exp` 向量指令（dav_c310 向量管 FTZ 语义）。
- **验证过的平台与 CANN 版本**：Ascend950（`__NPU_ARCH__==3510` → dav_c310），CANN cann-9.0.0-beta.2。
- **适用粒度**：`platform_wide`（Ascend950/dav_c310 向量管的 FTZ 行为，影响所有产生次正规 exp 结果的算子）。
- **路线优先级**：`banned`（硬件 FTZ 不可软件修复，须声明为硬件行为豁免交用户确认；禁止自行宣布"软件无法修复"，须先按 golden 运算顺序重算验证确实无法对齐）。
- **不适用/未验证场景**：`grad` 远离 FLT_MAX 的常规输入（次正规 exp × FLT_MAX 才可见发散；常规输入即使 FTZ 也无可见差异）；a2/a3 等其他平台的 FTZ 行为未验证。

## 失败路线

本卡为 `negative_result`：尝试在 kernel 侧消除全部精度漂移，但 3 个用例触及硬件 FTZ 边界，objective 不可达成。

- **尝试**：依次修复四类可修软件根因（Select SELMODE、L2 缓存键、计算顺序、`-50` 下界 clamp），使 PyTorch ST 失败数逐轮收敛至 3。
- **objective 失败**：收敛后仍有 `L1_018`/`L1_083`/`L1_281` 三例稳定失败，全部满足 `grad=±FLT_MAX` 且 `z∈[-104,-87]`（`exp(z)` 次正规）。计算顺序已匹配 aten（`grad*exp/(1+exp)`），`-50` clamp 已移除让 exp 自然下溢，但硬件 FTZ 仍把次正规 `exp` 输出刷新为 0，`npu=0` ≠ `golden≈6e-2`，无法在 kernel 侧消除。
- **边界**：仅当 `grad=±FLT_MAX` 且 `z∈[ln(FLT_MIN), 0)` ≈ `[-87.3, 0)`（`exp(z)` 为次正规但非零）时发散可见；`z < -87.3` 且 `exp(z)` 完全下溢为 0 时 golden=0（CPU 也下溢，对齐）；`z > 0` 时 `exp(z)` 为正常数（无 FTZ）。故仅 `z∈[-104,-87]` 窄带可见发散。
- **banned**：不尝试软件 workaround（如自行实现保留次正规的 exp）；向量管 inherently FTZ，`AscendC::Exp` 不提供次正规保留模式，kernel 无法绕过该刷新。声明为硬件 FTZ 豁免，交用户确认。

## 验证信号

- `L1_018`/`L1_083`/`L1_281` 三例稳定失败：`grad=±FLT_MAX`、`z∈[-104,-87]`、`exp(z)` 为次正规数（`1.76e-40 < FLT_MIN=1.175e-38`），golden 保留次正规（`≈6e-2 = FLT_MAX × 1.76e-40`）、npu=0（FTZ）（来源平台：950；证据等级：tool_observed）
- 全量 PyTorch ST 终态 575/578（3 fail 为本硬件 FTZ 边界）（来源平台：950；证据等级：tool_observed）
- C++ ST 57/57 通过（C++ ST 用例不触发此极端组合）（来源平台：950；证据等级：tool_observed）
- L0 功能用例 51/51 通过（来源平台：950；证据等级：tool_observed）

## 失效边界

- 仅 Ascend950/dav_c310 实测；a2/a3 等其他平台的向量管 FTZ 行为未验证，推断同样 FTZ（矢量单元通用语义）但未实测，可迁移性未确认。
- 仅 `grad=±FLT_MAX` 且 `z∈[-104,-87]` 窄带可见发散；常规输入无可见差异，本边界不触发。

## 预防

- 对 `grad` 接近 `±FLT_MAX` 的极值用例，若 `z` 落在次正规 exp 带（`[-104,-87]`），预期 NPU 与 CPU golden 在该窄带发散——属硬件 FTZ，声明为硬件行为，不追逐软件修复。
- 设计极值测试用例时，避免 `grad=±FLT_MAX` 且 `z` 恰落在次正规 exp 带的组合（该组合的 golden 本身依赖 CPU 保留次正规的实现细节，可移植性差）。