---
type: "Runbook"
title: "exp 下界 clamp -50 破坏 sigmoid 大负数下溢"
description: "SoftplusV2Grad kernel 对 exp 输入做 max(bx,-50) 下界 clamp，把大负数 z 的 exp(z)→0 正确下溢值抬高为 exp(-50)≈1.9e-22，gradOutput 大时被放大；判别：NPU 输出≈gradOutput×exp(-50) 即命中下界 clamp bug。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.activation", "topic.compute.elementwise.math_function"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# exp 下界 clamp -50 破坏 sigmoid 大负数下溢

## 现象

在 Ascend950PR 上为 `softplus_v2_grad` 做精度回归（golden=`torch.ops.aten.softplus_backward`，fp32 atol=1.22e-4/rtol=1.22e-3）。部分 L1 case 失败，特征为 `self`（即 `z=beta*self`）为大负数且 `gradOutput` 幅值很大时，NPU 输出远大于 golden=0。

最小复现：`z = beta*self = -1105`，`gradOutput = 3.4e38`（接近 FLT_MAX）。golden = `gradOutput * sigmoid(z) = 3.4e38 * 0 = 0`（sigmoid 在大负数处下溢为 0）。NPU 输出 `6.56e16`，正好等于 `gradOutput × exp(-50) ≈ 3.4e38 × 1.9e-22 ≈ 6.5e16`——即 exp 输入被 clamp 到 -50 后，`exp(-50)≈1.9e-22` 本应下溢为 0，却被人为抬高，再乘大 gradOutput 放大成可见误差。

## 适用场景

- **工程形态**：kernel 直调（RegBase，op_kernel/arch35 路径）。
- **API 路径**：`AscendC::Maxs`/`AscendC::Mins`（逐元素标量 clamp）/`AscendC::Exp`/`AscendC::Div`/`AscendC::Mul` 构成的 sigmoid 分支。
- **验证平台与版本**：Ascend950PR，CANN cann-9.0.0-beta.2。
- **适用粒度**：`operator_family`——任何含 `sigmoid/exp` 分支且对 exp 输入做下界 clamp 的算子（softplus 家族、sigmoid 相关 grad）。
- **路线优先级**：`recommended_fastpath`——移除无数学必要性的 exp 下界 clamp。
- **不适用/未验证场景**：上界 clamp（防 exp 正向溢出）必须保留，仅下界 clamp 是 bug；非 exp 类激活未验证；a2/a3 推断同 V pipe exp 语义但未实测，采用前需在目标平台复现探针确认。

## 判别方法

### trigger

精度回归中失败 case 集中在 `z=beta*self` 为大负数（如 `z<-50`）且 `gradOutput` 幅值大时；NPU 输出 ≈ `gradOutput × exp(-50)` 而非 0。最小复现 `z=-1105, gradOutput=3.4e38`：golden=0，NPU=6.56e16。

### probe

取单元素 `z=-1105, gradOutput=3.4e38` 跑 kernel，比对 NPU 输出与 `gradOutput × exp(-50) ≈ 6.5e16`：

- `NPU ≈ 6.5e16 = gradOutput × exp(-50)` → 命中下界 clamp bug；
- `NPU = 0`（或 fp32 非规范数）→ 非本 bug。

### observation

`z=-1105, gradOutput=3.4e38` 实测：golden=0，NPU=6.56e16≈`gradOutput × exp(-50)`。该值正是 exp 输入被 clamp 到 -50 后 `exp(-50)≈1.9e-22` 再乘大 gradOutput 的产物，与下界 clamp 抬高正确下溢值的机制吻合。

### interpretation

`ComputeFp32Core` 中对 exp 输入做双向 clamp（before 机制 artifact，read 自 `op_kernel/arch35/softplus_v2_grad_kernel.h` 的 `ComputeFp32Core` exp 输入 clamp 段）：

`bx_clamped = min(max(bx, -50.0), threshold)`

机制链：
- sigmoid 分支处理 `z <= threshold` 的元素。对 `z` 为大负数（如 -1105），数学上 `exp(z) → 0`（fp32 下正确下溢），`sigmoid(z) = exp(z)/(1+exp(z)) → 0`，`gradOutput × sigmoid(z) → 0`。
- `max(bx, -50)` 把 `bx` 从 -1105 抬到 -50，于是 `exp(bx_clamped) = exp(-50) ≈ 1.9e-22`（而非 0）。
- `gradOutput × 1.9e-22`：当 gradOutput ≈ 3.4e38 时，`3.4e38 × 1.9e-22 ≈ 6.5e16`，远超容差。
- 下界 clamp 无数学必要性：exp 对负输入只会单调趋于 0，不会溢出；上界 clamp（`Mins(bx, threshold)`）才是防 exp 正向溢出（`exp(88)` 已接近 fp32 上限）。

根因是对 exp 函数溢出方向的误解：exp 仅对正输入溢出，对负输入只会下溢到 0（正确值）；为"防溢出"加双向 clamp 时，下界 clamp 反而破坏了 sigmoid 大负数下溢的正确结果。

## 验证信号

本卡信号来自本次会话在 Ascend950PR 上的实测（CANN cann-9.0.0-beta.2）。

- 最小复现 `z=-1105/gradOutput=3.4e38`：golden=0，NPU=6.56e16≈`gradOutput×exp(-50)`（来源平台：950；证据等级: tool_observed）。
- before 机制 artifact：read 命中 `ComputeFp32Core` 内 `Maxs(bx,-50)+Mins(bx,threshold)` 双向 clamp（来源平台：950；证据等级: artifact_verified）。
- 应用修复（移除 `Maxs` 下界 clamp 行，仅保留 `Mins` 上界）后 PyTorch ST 失败由 114 降至 7（来源平台：950；证据等级: tool_observed）。
## 失效边界

- z∈[-95,-38] 区间 `exp(z)` 为 fp32 非规范数，NPU V pipe FTZ 会刷为 0；当 `gradOutput=±FLT_MAX` 时，golden（CPU 保留非规范数）得小非零值、NPU=0，仍不可对齐——这是硬件 FTZ，非本卡 bug，见极端值独立卡。
- a2/a3 推断可迁移（同 V pipe exp 语义），未验证，采用前需在目标平台复现探针确认。

## 预防

- 对 exp/log/sigmoid 类算子做输入 clamp 前，先分析函数在负/正方向的真实溢出方向：exp 仅正方向溢出，负方向下溢到 0（正确值）；不要为"防溢出"加双向 clamp。
- clamp 边界值（如 -50）的选取必须有数值依据，不能凭直觉。

## 坏实践

- "exp 输入双 clamp 防 exp 溢出"——下界 clamp 破坏 sigmoid 大负数下溢，是数值稳定性误区。