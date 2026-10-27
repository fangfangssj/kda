---
type: "Runbook"
title: "SoftplusV2Grad VF Form B exp 溢出判别"
description: "arch35 SoftplusV2Grad VF 链在 bf16/fp16 用 Form B(exp(cbx)/(1+exp(cbx))) 实例化时，大 threshold 或 grad=inf 使正向 Exp(cbx) 溢出为 inf/NaN 发散于 golden Form A 的判别：触发=mare 超阈、探针=sigmoid 分支形态、观测=22 FAIL→3 FAIL、解释=正向 exp 输入溢出。"
tags: ["task.debugging", "topic.compute.elementwise.activation", "topic.compute.elementwise.math_function", "topic.programming_model.execution_unit.vector", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T08:42:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["FormB", "FormA", "exp_overflow"]
---

# SoftplusV2Grad VF Form B exp 溢出判别

## 适用场景

- 工程形态：kernel 直调（arch35 `__simd_vf__` + `asc_vf_call` 连续 Vector 计算链），SoftplusV2Grad 反向梯度 `res = where(β·self > θ, g, g·sigmoid(β·self))`，VF 在 bf16/fp16 实例化时触发。
- API 路径：`AscendC::Reg::Muls/Mins/Exp/Adds/Div/Mul/Compares/Select` 寄存器级 API，封装在 `__simd_vf__ inline void SoftplusGradVF` 内、由 `asc_vf_call<SoftplusGradVF<T>>` 调用。
- 验证过的平台与 CANN 版本：Ascend950（dav-3510 / arch35），CANN 9.0.0-beta.2。仅本平台实测。
- 适用粒度：`operator_family`（适用于所有含 sigmoid/exp 反向梯度、在 bf16/fp16 Vector 寄存器内做 exp 的逐元素算子）。
- 路线优先级：`diagnostic_only`——本卡只提供 Form B 正向 exp 溢出的判别方法（触发/探针/原始观测/根因解释），不含 verbatim 修复 after（Form A 函数体可核于 trace seq 2215）。
- 不适用或未验证场景：fp32 VF 链（exp 输入上界 ~88.7，常规 θ 不溢出，属另一漂移族，见 [SoftplusV2Grad 非默认θ线性区分支缺失致超差卡](softplusv2grad.md)）；a2/a3 未验证（见 `## 失效边界`）。

## 判别方法

触发（tool_observed）：在 arch35 SoftplusV2Grad device-side VF 链（bf16/fp16 实例化）上跑 TTK 黑盒（golden_plugin 走 numpy 独立 Form A 参考），22 例 `mare` 超阈 46×–96×（相对各自 dtype 阈值），`mere<1e-7` 仅 outlier 元素超阈；均为 bf16/fp16、`threshold` 较大（9.17 / 17.67 / 18.13）或 `grad_output` 含 inf 的场景。同一 kernel 在 ST（golden 用 `torch.ops.aten.softplus_backward`）下 72/72 PASS——ST 与 TTK golden 都走 Form A，两侧在数值边界发散指向 kernel 端的 Form B。

探针：读 VF 链 sigmoid 分支的实现形态——若为 `e=Exp(cbx); r=Div(e, Adds(e,1)); pr=Mul(g, r)`（Form B，exp 输入是正向 `cbx`），且 `cbx=Mins(bx, θ)` 在大 `θ` 下仍是大正值，即命中；对 bf16/fp16 大 `threshold` 或 `grad=inf` 用例跑 TTK 黑盒读 `mere/mare`（floor=1e-7）确认 outlier 超阈。

原始观测（tool_observed，TTK 黑盒 result 汇总，取自 trace seq 994 tool_result）：

- 第 7 轮（Form B）：TTK 22 例 mare 超阈 46×–96×，FAIL；ST 72/72 PASS。
- 第 8 轮（Form B→Form A 后）：TTK 由 22 例 FAIL 降至 3 例（19 例转 PASS），538 PASS/3 FAIL；ST 72/72 不退化。

判别与解释（interpretation，agent 根因分析，非 API 规范主张）：VF 链用 Form B 实现 sigmoid `exp(cbx)/(1+exp(cbx))`，exp 输入是正向 `cbx`：

- 大 threshold 场景：`bx` 接近 `θ`（如 θ=17.67）时 `cbx` 仍是大正数，`Exp(cbx)` 在 fp16（exp 输入 >11 即溢出）/bf16（>88）溢出为 `+inf`，`r=inf/inf=NaN`、`pr=g·NaN=NaN`，发散于 golden Form A。
- g=inf 场景：`grad_output=inf` 配中等正 `bx`，Form B 的 `exp(cbx)` 已先溢出，`pr=inf·NaN=NaN`，与 golden Form A `g/(1+exp(-cbx))=inf/有界正数=inf` 不符。
- golden Form A `pr=g/(1+exp(-cbx))`：正向 `cbx` 大→`exp(-cbx)→0`→`denom→1`→`pr→g`（安全）；负向 `cbx` 大负→`exp(-cbx)→+inf`→`g/inf→0`（inf 算术下安全）。两向都不依赖正向大 exp，故不溢出。Form B→Form A 后 19 例转 PASS 印证正向 exp 溢出为根因。

机制 artifact（verbatim，Form B `SoftplusGradVF<T>` 计算链核心步，取自 trace seq 2377 tool_result，函数行 392–447，artifact_verified；本段为被诊断的 before 态，Form A 修复 after 未在本卡复述）：

```cpp
// SoftplusGradVF (Form B, 被诊断态)
AscendC::Reg::Muls(bxReg, xReg, beta, preg);              // S3: bx = β·x
AscendC::Reg::Mins(cbxReg, bxReg, threshold, preg);       // S4: cbx = min(bx, θ)
AscendC::Reg::Exp(eReg, cbxReg, preg);                    // S5: e = exp(cbx)  ← 正向 exp，大 cbx 溢出
AscendC::Reg::Adds(dReg, eReg, static_cast<T>(1.0), preg);// S6: d = 1 + e
AscendC::Reg::Div(rReg, eReg, dReg, preg);                 // S7: r = e / d = sigmoid(cbx)
AscendC::Reg::Mul(prReg, gReg, rReg, preg);               // S8: pr = g · r
AscendC::Reg::Compares<T, AscendC::CMPMODE::GT>(maskReg, bxReg, threshold, preg); // S9: mask=(bx>θ)
AscendC::Reg::Select(resReg, gReg, prReg, maskReg);        // S10: res = mask ? g : pr
```

结论：bf16/fp16 VF 链上 Form B 的正向 `Exp(cbx)` 在大 `threshold`/`grad=inf` 用例溢出为 `inf/NaN`，是与 golden Form A 发散的根因；判别只须确认 sigmoid 分支用的是 Form B（exp 输入正向 `cbx`）且 dtype 的 exp 输入上界低于 `θ` 取值。该根因解释属 agent 推断，hex 级 NaN payload 未在 tool_result 直接捕获。

## 验证信号

来源：本 trace 在 Ascend950（dav-3510 / arch35，CANN 9.0.0-beta.2）上对 `operators/SoftplusV2Grad_package/op_kernel/arch35/softplus_v2_grad_kernel.h` 的 VF 链跑 TTK 黑盒（numpy Form A golden_plugin）+ ST 回归；VF 链函数体作为 tool_result 完整逐字可核。

- TTK 黑盒 golden_plugin 22 例 `mare` 超阈 46×–96×（bf16/fp16 大 threshold / grad=inf），`mere<1e-7` 仅 outlier 超阈，FAIL（来源平台：950；证据等级：tool_observed）
- Form B→Form A 改写后 TTK 由 22 例 FAIL 降至 3 例（19 例转 PASS），538 PASS/3 FAIL（来源平台：950；证据等级：tool_observed）
- ST（torch golden）72/72 回归不退化（Branch-0 43 + Branch-1 29，oracle 独立于被测 kernel）（来源平台：950；证据等级：tool_observed）
- Form B `SoftplusGradVF<T>` 计算链（S5 `Exp(cbx)` 正向 exp）作为 tool_result 逐字可核，绑定 trace seq 2377（来源平台：950；证据等级：artifact_verified）
- `asc_vf_call`(6 处) / `__simd_vf__`(3 处) 满足 dav-3510 连续 Vector 计算链硬要求（来源平台：950；证据等级：artifact_verified）

## 失效边界

- Ascend950（dav-3510 / arch35）/ CANN 9.0.0-beta.2、bf16/fp16/fp32 实测有效；仅覆盖 sigmoid-导数类算子 VF 链在 bf16/fp16 实例化时的正向 exp 溢出。
- fp32 VF 链不适用：fp32 exp 输入上界 ~88.7，常规 `θ`(<88) 下 `Exp(cbx)` 不溢出，属另一数值漂移族（见 [SoftplusV2Grad 非默认θ线性区分支缺失致超差卡](softplusv2grad.md)的四区漂移）。
- a2/a3 平台 `Reg::Exp` 的输入上界与舍入未验证；Form A 是纯数学等价改写，推断可迁移，但 exp 输入上界随硬件版本变化，迁移后须重测 `θ` 与 dtype 的溢出边界。
- `θ` 取极大值（bf16 接近 88、fp16 接近 11）时即便 Form A 的 `exp(-cbx)` 输入 `-cbx` 仍可能达 `-θ`，若 `θ` 超过 dtype 的 exp 输入上界 Form A 仍会溢出——须在上游约束 `θ` 取值域或加额外 clamp。

## 坏实践

- 用 Form B 后为过测试而放宽容差（rtol/atol 调松到恰好通过）——掩盖 exp 溢出真因。
- 用高精度 Exp（`PRECISION_1ULP_FTZ_FALSE`）硬扛 Form B 溢出——致 RANK_8 编译失败，且不解决 g=inf 时 inf·NaN 发散。
- 以 ST（torch golden）PASS 反证 VF 链无溢出——ST golden 也走 Form A，与 kernel 的 Form B 在数值边界同样发散，PASS 只表示与同形态 golden 一致。