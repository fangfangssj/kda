---
type: "Runbook"
title: "exp 输入 clamp 上界用 threshold 用户参数致溢出"
description: "arch35 kernel 用算子用户参数 threshold 作 exp 输入 clamp 上界，threshold 可达 3.4e38 使 exp(100)→inf→Actual=3.4e38；clamp 上界改固定 50.0f 与 threshold 解耦修复。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.math_function"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# exp 输入 clamp 上界用 threshold 用户参数致溢出

## 现象

Ascend950PR 上 SoftplusV2Grad arch35 kernel 的 PyTorch ST NPU 验收中，16 例输出溢出：如 `[69] fp32 mismatch=320100/441000 Golden=0 Actual=3.403e+38`。Golden=0（gradOutput=0 且应走 sigmoid 分支输出 0），但 Actual 是 fp32 最大值，说明 `exp` 计算溢出到 inf 后又被某种路径规约为 3.4e38。失败只出现在 CSV 驱动的极端值域用例（threshold/beta/self 取天文数字），C++ ST 的固定值域不触发。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发，arch35 kernel 直调，PyTorch ST NPU 批量验收。
- **API 路径**：`AscendC::Mins` / `AscendC::Maxs`（clamp）+ `AscendC::Exp`。
- **验证平台与版本**：Ascend950PR（3510/DAV_3510，平台 token `950`），CANN cann-9.0.0-beta.2。
- **适用粒度**：operator_family（任何用 user-supplied threshold/上限参数去 clamp exp/log/reciprocal 等易溢出函数输入的 arch35/950 kernel）。
- **路线优先级**：recommended_fastpath（clamp 上界用固定安全常数，与用户参数解耦，零代价）。
- **不适用或未验证场景**：threshold 恒为小常数（如 ≤50）的算子不会触发；非 exp 类易溢出函数（log/reciprocal）的 clamp 边界需各自推导安全上界，不能照搬 50。

## 根因

`ComputeFp32Core` 用 PyTorch 等价形式 `sigmoid(z)=exp(z)/(1+exp(z))`（规避大负数 exp(-z) 溢出），对 exp 输入做了双 clamp：`bx_clamped = min(max(bx, -50), threshold)`。**上界用了算子用户参数 `threshold`**。`threshold` 是 SoftplusV2Grad 的输入参数，CSV 极端用例可取 3.4e38。

关键推导：当 `bx ≤ threshold`（走 sigmoid 分支）时，`bx_clamped = min(max(bx, -50), threshold) = min(bx, threshold) = bx`（因 bx 已 ≤ threshold）。所以 exp 的输入就是原始 bx——上界 clamp 形同虚设。若 `threshold=3.4e38` 且 `bx=100`（bx ≤ 3.4e38 走 sigmoid），`bx_clamped=100`，`exp(100)=2.69e43` 在 fp32 中溢出到 inf，`sig=inf/(1+inf)=inf/inf=NaN`，后续传播为 3.4e38。

注释写"防 exp 溢出"但上界用 threshold，而 threshold 本身就是可能取天文数字的用户参数——clamp 上界与被防的量同源，等于没防。正确上界应是固定安全值：`exp(50)=5.18e21` 不溢出 fp32（fp32 max ≈ 3.4e38）。

## 解决配方

修复点 `softplus_v2_grad_kernel.h:214`。before（clamp 上界 = threshold 用户参数）：

```cpp
AscendC::Mins<float>(bBx, bBx, threshold_, count);  // clamp 上界 = threshold
```

after（clamp 上界固定 50.0f，与 threshold 解耦）：

```cpp
AscendC::Mins<float>(bBx, bBx, 50.0f, count);  // 固定上界，防 exp 溢出
```

同步更新文件头注释（verbatim edit，adapter 未截断）：

before:
```
 *   bx_clamped = min(max(bx, -50), threshold)   // exp 输入双 clamp 防 exp 溢出
```
after:
```
 *   bx_clamped = min(max(bx, -50), 50.0)    // exp 输入双 clamp 防 exp 溢出（上界固定 50，与 threshold 解耦）
```

数学公式自洽性：spec.yaml 真值源 `gradInput = gradOutput × sigmoid(z)`，`sigmoid(z)=1/(1+exp(-z))`。等价变形 `exp(z)/(1+exp(z))` 在 `|z|≤50` 内与原式 bit-exact 一致；`|z|>50` 时 sigmoid 本就饱和到 0 或 1，clamp 到 50 不改变数学结果，只消除 inf/NaN 传播路径。下界 -50 对称（`exp(-50)≈0`，sigmoid≈0）。threshold 分支判断仍用原始 bx（未 clamp），不受影响。

> 注：完整 `ComputeFp32Core` 函数体 verbatim 终版写入事件被 adapter 截断，上方仅内联 agent 在 reasoning 事件中原样引用的关键行 + 未截断的注释 edit。完整函数级 before/after 未完全从 trace 恢复—。

## 预防

- 任何对易溢出函数（exp/log/reciprocal/pow）输入做的 clamp，上下界必须用**与输入无关的固定安全常数**，禁止用算子用户参数（threshold/beta/alpha 等可被外部赋天文值的量）作 clamp 边界。
- DESIGN_PREP 的精度策略审查应明确问："这个 clamp 边界值的取值范围是多少？它是常数还是用户参数？若用户参数取 3.4e38 会怎样？"
- C++ ST 的值域要覆盖 threshold/beta 取极端值（0/1/inf/nan/3.4e38）的 case，否则 CSV 驱动的边界失败不会在 L0 暴露。

## 验证信号

修复后（commit `0cefa32`，与本库 DRAFT-precision-01 同 commit 提交）：
- PyTorch ST NPU 16 例溢出类失败清零，总失败由 120 降至 58（来源平台：950；证据等级: tool_observed）
- C++ ST 真机 57/57 无回归（来源平台：950；证据等级: tool_observed）
- 编译通过（来源平台：950；证据等级: tool_observed）
## 性能影响

无相对性能代价（参照路线：原 threshold 上界，来源平台：950）——Mins 指令不变，仅常数操作数不同。

## 失效边界

- 950PR（3510）已实测；950DT 推断可迁移，未验证。
- 2201 等其它架构的 fp32 exp 溢出阈值相同（IEEE fp32），clamp 到 50 的安全边界通用；但若 kernel 用 bf16/fp16 exp，须重新推导安全上界（fp16 max≈6.5e4，exp(11.1)≈6.5e4，上界要更小）。
- 下界 -50 同理：bf16/fp16 须收紧。

## 坏实践

- 注释声明"防 exp 溢出"但 clamp 上界用 user-supplied threshold——clamp 边界与被防的溢出来源同源，等于没防。
- C++ ST 只用固定小值域（threshold=20）就声明精度通过，未覆盖 user 参数极端值域。