---
type: "Runbook"
title: "fp16 大累加项数跨实现 MERE 越阈边界"
description: "fp16 输出收缩算子每个输出元素累加 32768 个乘积（fp32 acc 单次舍回）时，相对 PyTorch aten golden 的 MERE=1.36e-3 超过 fp16 阈值 9.77e-4（累加 2048/4096 项时为 1.3e-6/3e-4 量级）；同实现 golden 的 C++ ST MERE=0.00 背书逻辑正确。大累加项数是该实现路线的跨实现精度边界，处置=登记边界并保留用例作回归哨兵。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# fp16 大累加项数跨实现 MERE 越阈边界

## 现象

为一个 AscendC 自定义算子（Conv3D filter 梯度反向，fp16 输入输出，Ascend950PR）跑 PyTorch ST 全量用例：小/中 batch 与小 shape 用例全部通过（单 batch 用例 MERE=1.3e-6，双 batch 的 L0 用例 MERE≈2.5e-4~3.3e-4，最小 shape 用例 MERE=0.00），唯独 batch 上限用例（x [16,64,8,16,16]、grad [16,128,8,16,16]，每个输出元素需累加 16×8×16×16 = 32768 个 fp16 输入乘积）失败：

```
[N=16 batch上限边界]
  x: [16,64,8,16,16], grad: [16,128,8,16,16], K: 3x3x3, groups=1, dtype=torch.float16
    MERE: 1.364652e-03 (阈值: 9.765625e-04) FAIL
    MARE: 3.150024e-01 (阈值: 9.765625e-03) FAIL
    最大相对误差位置: [93447] Golden=0.241089, Actual=-0.073914
```

而同一算子、同一批用例语义的 C++ ST（golden 用与 kernel 相同的逐元素累加实现）全部 MERE=0.00——算子逻辑本身正确，超阈只发生在与 PyTorch golden 的跨实现比对、且累加项数最大的用例上。

## 适用场景

- 工程形态：算子精度验收（PyTorch golden 跨实现比对）与精度边界登记。
- API 路径：kernel 标量 contraction（float acc 累加后单次 `static_cast<D_T_X>(acc)` 舍回 fp16 输出）；PyTorch `F.grad.conv3d_weight` golden。
- 验证过的平台与 CANN 版本：Ascend 950PR（ascend950），CANN 9.1.0。
- 适用粒度：operator_family（收缩/累加类 fp16 输出算子的"累加项数 × 输出舍入"边界）。
- 路线优先级：safety_fallback（当前无 workspace 标量路线在大累加项数场景记为精度受限并登记；向量 Cast + fp32 累加/分块归约等替代路线未验证）。fallback 声明：接受该边界意味着大 batch 用例不进入精度验收通过项，需在验收报告中显式登记。
- 明确不适用 / 未验证场景：bf16 及其他 dtype；fp32 输出算子；替代累加组织（分块/树形归约）未验证。

## 根因

已锚定的机制事实（代码与观测可证）：kernel 对每个输出元素以 `float acc = 0.0f` 累加 32768 个 `static_cast<float>(fp16)*static_cast<float>(fp16)` 乘积后，单次 `static_cast<D_T_X>(acc)` 舍回 fp16 输出（op_kernel/arch35 头文件读回实录：`float acc = 0.0f; … acc += gv * xv; … outUb.SetValue(outIdx, static_cast<D_T_X>(acc));`）；PyTorch golden 走 aten::convolution_backward 的另一套累加组织。观测边界：累加 2048 项（单 batch）MERE=1.3e-6，4096 项（双 batch）MERE≈3.1e-4，32768 项（batch 上限）MERE=1.36e-3 超阈，且最大偏差出现在小数值输出位置（Golden=0.241 vs Actual=-0.074，偏差 0.315）——大量乘积累加后经输出舍入，两种累加组织在小数值输出处产生可观分歧。机制层定量归因（fp16 输入量化误差、累加顺序差异、输出舍入三者各占多少）未完成，本卡只主张"累加项数增大到 32768 时跨实现偏差越过 fp16 阈值"这一观测结论，不主张更细的误差模型。

## 解决配方

无代码修复（当前路线下的已验证处置是登记边界）：

- 精度验收报告中把该用例登记为"部分通过 + 原因"，并在算子 README 约束说明中写明：大 batch（大累加项数）场景精度受限，建议大 batch 场景减小 N 或改用向量 Cast + fp32 累加方案。
- 测试用例保留该用例（不删除），结果如实计 FAIL，作为边界回归哨兵。
- 替代路线（向量 Cast API + workspace 中转的 fp32 累加，或分块归约）在当前架构（arch35 无 workspace 直接计算方案）下不可用/未实现，未验证。

## 验证信号

来源说明：以下信号来自 PyTorch ST（NPU 实测，golden 在 CPU）与 C++ ST 的对照运行（Ascend950PR，CANN 9.1.0）。

- 边界观测：batch 上限用例 `MERE: 1.364652e-03 (阈值: 9.765625e-04) FAIL`，`最大相对误差位置: [93447] Golden=0.241089, Actual=-0.073914`（来源平台：950；证据等级：tool_observed）
- 对照组同轮实测：单 batch `MERE: 1.314359e-06 PASS`、最小 shape `MERE: 0.000000e+00 PASS`、双 batch 各 L0 用例 MERE 2.5e-4~3.3e-4 PASS（来源平台：950；证据等级：tool_observed）
- 逻辑正确性背书：C++ ST Real（golden 与 kernel 同实现累加）L0 5/5 `MERE=0.00e+00, MARE=0.00e+00`（来源平台：950；证据等级：tool_observed）
- kernel 累加组织证据：`float acc = 0.0f` / `acc += gv * xv` / `SetValue(static_cast<D_T_X>(acc))`（来源平台：950；证据等级：tool_observed）

## 失效边界

- 观测边界点：单输出元素累加 32768 项超阈；2048 项与 4096 项通过。中间规模未逐点测量，不能外推为线性规律（单随机种子单次观测）。
- 该边界绑定"PyTorch golden 跨实现比对"口径；与 kernel 同实现的 golden 口径下无此偏差（MERE=0.00）。
- 其他 shape/dtype/累加组织未验证；其他平台未实测，未验证，推断可迁移（fp16 表示极限与累加项数的关系与平台无关）。

**未决项**：

- **优先项：复现并定量分解**大累加项数偏差的误差来源（fp16 输入量化 × 累加组织差异 × 输出舍入），确定可否用分块/补偿累加在当前架构内恢复达标。
- 向量 Cast + fp32 累加（需 workspace 中转）路线在本架构下与无 workspace 方案冲突，其精度收益未验证。
## 坏实践

- 把该用例的 FAIL 通过放宽容差"修绿"——跨实现偏差是真实的数值事实，容差放宽会同时掩盖未来的真实回归。
- 只报 MERE 不报最大偏差位置——小数值输出处的 0.315 偏差才是定位线索。

## 误诊

- 当时叙述为"fp16 标量累加 32768 次精度损失累积"；代码读回证实累加器是 fp32，损失主要来自输入 fp16 量化与输出舍入在两种累加组织间的分歧，"fp16 累加"的表述不准确（推断表述已被代码实录修正）。

## 预防

- 收缩/累加类算子的测试设计显式包含"累加项数阶梯"用例（如单/双/8/16 batch 梯度），把精度边界在验收前测出并登记，而不是等用户场景踩到。
- 验收报告区分"逻辑正确性（同实现 golden）"与"跨实现数值一致性（PyTorch golden）"两条口径分别给结论。
