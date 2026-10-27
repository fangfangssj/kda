---
type: "Runbook"
title: "Abs 原地取绝对值后量化丢符号致 int8 坍缩"
description: "动态量化 kernel 先 Abs(yNormFp32, yNormFp32) 原地取绝对值再 Muls(1/scale) 量化，会丢失符号、量化的是绝对值，导致 int8 输出坍缩为 ±1 量级、约 50% mismatch；须把 Abs 写入独立 buffer、Muls 作用在原始带符号 tensor 上。"
tags: ["task.debugging", "topic.precision", "topic.compute.quantization", "topic.compute.elementwise.math_function"]
status: draft
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
aliases: ["sign_loss"]
---

# Abs 原地取绝对值后量化丢符号致 int8 坍缩

## 现象

`add_rms_norm_dynamic_quant` 算子 Kernel 首次编译通过后，运行精度对比脚本（NPU 输出 vs FP64 参考实现，逐 shape 打印 `int8_match`/`mismatch`/`xOut_diff`/`scale_diff`）。现象：`xOut`（Add 输出）与 `scaleOut`（per-token scale）两个输出全部精确匹配、`xOut_diff=0.000000`、`scale_diff≈0`，唯独 int8 输出 `y` 大面积不匹配——多个 shape 的 mismatch 在 0.46~0.94 之间：

```
FAIL: M=   1, N=  16, dtype=torch.float16        | int8_match=False, mismatch=0.9375 | xOut_diff=0.000000, close=True | scale_diff=0.00000000, close=True
FAIL: M=  16, N=4096, dtype=torch.float16        | int8_match=False, mismatch=0.4624 | xOut_diff=0.000000, close=True | scale_diff=0.00000001, close=True
FAIL: M= 128, N=4096, dtype=torch.bfloat16       | int8_match=False, mismatch=0.4649 | xOut_diff=0.000000, close=True | scale_diff=0.00000001, close=True
```

进一步单 case 打印：NPU 的 `y` 值非常小（0, 1, -1, 2 量级），而参考值分布在完整 int8 范围（如 `y_ref: [16, 59, -67, -50, -45, -82, 127, ...]` 对比 `y_npu: [16, 1, -1, 1, 1, -1, -1, 0, ...]`，第 0 个恰好碰巧相同）。`scale` 正确意味着 `max(|y_norm|)` 与 `invScale=127/maxAbs` 正确，但 `y` 没被正确放大。当时 Kernel 代码处于编译刚通过、量化步骤用 `Abs(yNormFp32, yNormFp32)` 原地取绝对值的状态。

## 适用场景

- **工程形态**：kernel 直调，FP32 中间计算 + per-token 对称动态量化到 int8。
- **API 路径**：AscendC `Abs`/`Muls`/`ReduceMax`，UB 上 FP32 tensor 的量化流水线。
- **验证平台与版本**：ascend910b（dav-2201），CANN 9.0.0。不在注册平台集合 {950,a3,a2}，`platforms` 为空表。
- **适用粒度**：`operator_family`（任何“先取绝对值求 scale、再量化带符号原值”的动态量化算子）。
- **路线优先级**：`recommended_fastpath`（Abs 写独立 buffer，量化原值）。
- **不适用/未验证场景**：scale 仅由绝对值决定、输出本就是绝对值（无符号）的算子不受此坑；FP32 vs FP64 边界 ±1 残差不在本卡范围。

## 根因

量化步骤写成“先 `Abs(yNormFp32, yNormFp32, N32)` 原地取绝对值，再 `ReduceMax` 求最大绝对值，再 `Muls(yNormFp32, yNormFp32, invScale)` 量化”。问题在第三步：`Abs` 是 in-place（dst==src），执行后 `yNormFp32` 已全部变成正数、符号信息丢失；随后 `Muls(yNormFp32, yNormFp32, invScale)` 量化的是这些绝对值，不是原始带符号的 `y_norm`。于是输出全是正数、且被 `invScale` 放大后又经 clamp 截断，绝大多数落在 ±1 量级，与参考的带符号全范围值对不上。机制链：`Abs(dst=src)` 覆写原值 → 后续以同一 tensor 做 `Muls` 时源已是绝对值 → 量化丢符号 → int8 坍缩。正确做法是：`scale` 用 `|y_norm|` 的最大值算，但 `1/scale` 必须乘在**原始带符号** `y_norm` 上。

## 解决配方

修复前（`ProcessOneRow` 量化步骤，Step 5a-5d；`Abs` 原地后 `Muls` 量化的是绝对值）：

```cpp
        // ===== Step 5: Quantization =====
        // 5a: Abs(yNormFp32) in-place
        AscendC::Abs<float>(yNormFp32, yNormFp32, N32);

        // 5b: ReduceMax → maxAbsVal
        AscendC::ReduceMax<float>(scalarBuf, yNormFp32, reduceTmp, N32, false);
        float maxAbsF = scalarBuf.GetValue(0);

        // 5c: scaleOut = maxAbs / 127.0
        float scaleOutVal = maxAbsF / 127.0f;

        // Store scaleOut with stride 2 for 8B alignment
        auto scaleOutAccLocal = scaleOutAccBuf_.Get<float>();
        scaleOutAccLocal.SetValue(rowIdx * 2, scaleOutVal);

        // 5d: qVal = yNormFp32 * (1.0 / scaleOut)
        // Avoid division by zero: if scaleOut is 0, all values are 0
        float invScale = (scaleOutVal != 0.0f) ? (1.0f / scaleOutVal) : 0.0f;
        AscendC::Muls<float>(yNormFp32, yNormFp32, invScale, N32);
```

修复后（`Abs` 写入独立 `absBuf`（复用已写出的 `workBuf2_`=xOutFp32），`ReduceMax` 用 `absBuf`，`Muls` 量化原始带符号 `yNormFp32`）：

```cpp
        // ===== Step 5: Quantization =====
        // 5a: Compute |yNormFp32| into a temp buffer for ReduceMax
        // We reuse workBuf[2] (xOutFp32, already written out) for the abs values
        auto absBuf = workBuf2_.Get<float>();
        AscendC::Abs<float>(absBuf, yNormFp32, N32);

        // 5b: ReduceMax on abs values → maxAbsVal
        AscendC::ReduceMax<float>(scalarBuf, absBuf, reduceTmp, N32, false);
        float maxAbsF = scalarBuf.GetValue(0);

        // 5c: scaleOut = maxAbs / 127.0
        float scaleOutVal = maxAbsF / 127.0f;

        // Store scaleOut with stride 2 for 8B alignment
        auto scaleOutAccLocal = scaleOutAccBuf_.Get<float>();
        scaleOutAccLocal.SetValue(rowIdx * 2, scaleOutVal);

        // 5d: qVal = yNormFp32 * (1.0 / scaleOut) — using ORIGINAL yNormFp32 (with sign)
        // Avoid division by zero: if scaleOut is 0, all values are 0
        float invScale = (scaleOutVal != 0.0f) ? (1.0f / scaleOutVal) : 0.0f;
        AscendC::Muls<float>(yNormFp32, yNormFp32, invScale, N32);
```

## 设计决策

- `Abs` 不再 in-place，改为写入 `workBuf2_`（xOutFp32）：Step 4 已把 xOut 写回 GM，`workBuf2_` 在 Step 5a 时空闲，复用其 N×float 容量作 absBuf，避免新分配 UB。`Get<float>()` 只返回 tensor 视图、不是新分配，故先后复用同一 `workBuf2_` 作 absBuf 与（后续 clamp 的）广播 buffer 在顺序使用下安全。
- 量化（`Muls` 1/scale）必须作用在原始带符号 `yNormFp32`：scale 由 `|y_norm|` 决定，但量化对象是带符号原值，二者不能共用同一 in-place tensor。

## 预防

写“取绝对值求 scale、再量化原值”的流水线时，`Abs` 的 dst 必须独立于将被量化的源 tensor；in-place `Abs(dst=src)` 后若同一 tensor 还要被后续带符号运算使用，即丢符号。审查量化 kernel 时把“Abs in-place + 后续同 tensor 运算”作为可疑点。

## 验证信号

来源：本轮算子精度联调。修复后重测，大 N 的 mismatch 从约 0.46 降至约 0.018（M=16,N=4096,fp16：0.4624→0.0180），xOut/scale 保持精确匹配。小 N 仍高（M=1,N=16：0.9375）——此为另一独立根因（clamp 的 Max/Min 广播），本卡只主张“Abs 丢符号”这一根因的消除。（来源平台：未定位；证据等级: tool_observed）
## 失效边界

本卡仅覆盖“Abs in-place 丢符号致量化坍缩”。修复后小 N 仍存的 mismatch 属 clamp 的 Max/Min 广播与 buffer 问题，不在此卡范围。本 trace 验证平台为 ascend910b（dav-2201），FP32 中间计算路径推断在其它 AICore 微架构同行为——未验证，推断可迁移。

## 误诊

最初怀疑 `xOut`/`scale` 本身错，但实测二者 `xOut_diff=0.000000`、`scale_diff≈0` 全部精确匹配，排除上游；进一步怀疑 `Muls` 的 `invScale` 数值错，但 `scale` 正确即 `invScale=127/maxAbs` 正确，排除；最终定位到 `Abs` in-place 覆写带符号源。