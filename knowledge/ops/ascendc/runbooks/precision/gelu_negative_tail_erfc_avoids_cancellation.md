---
type: "Runbook"
title: "GELU 负尾部用 erfc 形态避免大数相消"
description: "fp32 exact GELU 用 0.5x 加减近等幅项会在负大值尾部相消；直接计算 max(x,0)-0.5|x|erfc(|x|/sqrt(2)) 后 20/20。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.activation", "topic.compute.elementwise.math_function"]
status: draft
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
---

# GELU 负尾部用 erfc 形态避免大数相消

## 现象

GELU exact 路线的 fp32 大值域用例持续失败：多数普通范围和 fp16/bf16 用例已通过，但包含很负输入的用例仍失败，MARE 约 0.855。进一步提高 `erf` 近似精度后失败仍集中在负尾部，说明误差不是简单的 `erf` 精度不足。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`GELU`、`erfc`、`erf`、`cancellation` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 未定位；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

唯一根因是公式形态在负尾部发生灾难性相消。`0.5*x + 0.5*|x|*erf(|x|/sqrt(2))` 中两项幅值都接近 `0.5|x|`、符号相反，而正确结果接近 0；即使 `erf` 近似已经较准，有限精度下两大项相消仍会放大残差。

## 解决配方

保留相同的 Hastings 系数、Reciprocal Newton 修正和 tanh 分支，只把 exact 分支改写为直接计算小尾部：

`y = max(x, 0) - 0.5 * |x| * erfc(|x| / sqrt(2))`

### before

`csrc/ops/gelu/op_kernel/gelu_kernel.cpp`，`ComputeFp32`，修复前：

```cpp
    __aicore__ inline void ComputeFp32(const AscendC::LocalTensor<float>& xF,
                                       const AscendC::LocalTensor<float>& yF,
                                       int64_t count)
    {
        auto t1 = tmpBuf_.Get<float>();
        auto t2 = tmpBuf2_.Get<float>();
        auto t3 = tmpBuf3_.Get<float>();

        if (mode_ == GELU_MODE_NONE) {
            // z = |x|/sqrt(2)  -> t2
            AscendC::Abs (t2, xF, count);
            AscendC::Muls(t2, t2, INV_SQRT2, count);
            // d = 1 + p*z  -> t3
            AscendC::Muls(t3, t2, AS_P, count);
            AscendC::Adds(t3, t3, 1.0f, count);
            // u = 1/d via Reciprocal + 2 Newton steps r' = r*(2 - d*r)
            AscendC::Reciprocal(t1, t3, count);
            AscendC::Mul (yF, t3, t1, count);
            AscendC::Muls(yF, yF, -1.0f, count);
            AscendC::Adds(yF, yF, 2.0f, count);
            AscendC::Mul (t1, t1, yF, count);
            AscendC::Mul (yF, t3, t1, count);
            AscendC::Muls(yF, yF, -1.0f, count);
            AscendC::Adds(yF, yF, 2.0f, count);
            AscendC::Mul (t1, t1, yF, count);
            // poly(u) via Horner -> t3
            AscendC::Muls(t3, t1, AS_A5, count);
            AscendC::Adds(t3, t3, AS_A4, count);
            AscendC::Mul (t3, t3, t1, count);
            AscendC::Adds(t3, t3, AS_A3, count);
            AscendC::Mul (t3, t3, t1, count);
            AscendC::Adds(t3, t3, AS_A2, count);
            AscendC::Mul (t3, t3, t1, count);
            AscendC::Adds(t3, t3, AS_A1, count);
            AscendC::Mul (t3, t3, t1, count);
            // t1 = exp(-z*z)
            AscendC::Mul (t1, t2, t2, count);
            AscendC::Muls(t1, t1, -1.0f, count);
            AscendC::Exp (t1, t1, count);
            // erf_z = 1 - poly*e  -> t3
            AscendC::Mul (t3, t3, t1, count);
            AscendC::Muls(t3, t3, -1.0f, count);
            AscendC::Adds(t3, t3, 1.0f, count);
            // y = 0.5*x + 0.5*|x|*erf_z
            AscendC::Abs (t1, xF, count);
            AscendC::Mul (t3, t3, t1, count);
            AscendC::Muls(t3, t3, 0.5f, count);
            AscendC::Muls(yF, xF, 0.5f, count);
            AscendC::Add (yF, yF, t3, count);
        } else {
            AscendC::Mul (t1, xF, xF, count);
            AscendC::Mul (t1, t1, xF, count);
            AscendC::Muls(t1, t1, K_TANH, count);
            AscendC::Add (t1, t1, xF, count);
            AscendC::Muls(t1, t1, SQRT_2_OVER_PI, count);
            AscendC::Tanh(t1, t1, count);
            AscendC::Adds(t1, t1, 1.0f, count);
            AscendC::Muls(t1, t1, 0.5f, count);
            AscendC::Mins(t1, t1, 1.0f, count);
            AscendC::Maxs(t1, t1, 0.0f, count);
            AscendC::Mul (yF, t1, xF, count);
        }
    }
```

### after

同文件、同 callable，修复后：

```cpp
    __aicore__ inline void ComputeFp32(const AscendC::LocalTensor<float>& xF,
                                       const AscendC::LocalTensor<float>& yF,
                                       int64_t count)
    {
        auto t1 = tmpBuf_.Get<float>();
        auto t2 = tmpBuf2_.Get<float>();
        auto t3 = tmpBuf3_.Get<float>();

        if (mode_ == GELU_MODE_NONE) {
            // Stable formulation: y = max(x,0) - 0.5*|x|*erfc(|x|/sqrt(2))
            // erfc(z) via Hastings (A&S 7.1.26): erfc(z) ~= poly(u) * exp(-z^2),
            // where u = 1/(1 + p*z), poly = a1*u + a2*u^2 + ... + a5*u^5.
            AscendC::Abs (t1, xF, count);
            AscendC::Muls(t2, t1, INV_SQRT2, count);
            AscendC::Muls(t3, t2, AS_P, count);
            AscendC::Adds(t3, t3, 1.0f, count);
            AscendC::Reciprocal(yF, t3, count);
            AscendC::Mul (t1, t3, yF, count);
            AscendC::Muls(t1, t1, -1.0f, count);
            AscendC::Adds(t1, t1, 2.0f, count);
            AscendC::Mul (yF, yF, t1, count);
            AscendC::Mul (t1, t3, yF, count);
            AscendC::Muls(t1, t1, -1.0f, count);
            AscendC::Adds(t1, t1, 2.0f, count);
            AscendC::Mul (yF, yF, t1, count);
            AscendC::Abs (t1, xF, count);
            AscendC::Muls(t3, yF, AS_A5, count);
            AscendC::Adds(t3, t3, AS_A4, count);
            AscendC::Mul (t3, t3, yF, count);
            AscendC::Adds(t3, t3, AS_A3, count);
            AscendC::Mul (t3, t3, yF, count);
            AscendC::Adds(t3, t3, AS_A2, count);
            AscendC::Mul (t3, t3, yF, count);
            AscendC::Adds(t3, t3, AS_A1, count);
            AscendC::Mul (t3, t3, yF, count);
            AscendC::Mul (yF, t2, t2, count);
            AscendC::Muls(yF, yF, -1.0f, count);
            AscendC::Exp (yF, yF, count);
            AscendC::Mul (t3, t3, yF, count);
            AscendC::Mul (t3, t3, t1, count);
            AscendC::Muls(t3, t3, 0.5f, count);
            AscendC::Maxs(yF, xF, 0.0f, count);
            AscendC::Sub (yF, yF, t3, count);
        } else {
            AscendC::Mul (t1, xF, xF, count);
            AscendC::Mul (t1, t1, xF, count);
            AscendC::Muls(t1, t1, K_TANH, count);
            AscendC::Add (t1, t1, xF, count);
            AscendC::Muls(t1, t1, SQRT_2_OVER_PI, count);
            AscendC::Tanh(t1, t1, count);
            AscendC::Adds(t1, t1, 1.0f, count);
            AscendC::Muls(t1, t1, 0.5f, count);
            AscendC::Mins(t1, t1, 1.0f, count);
            AscendC::Maxs(t1, t1, 0.0f, count);
            AscendC::Mul (yF, t1, xF, count);
        }
    }
```

## 设计决策

使用 `erfc` 直接表示负尾部概率，避免先构造接近 1 的 `erf` 再让两个大项相消。`max(x,0)` 对正侧保留主项，对负侧为 0；剩余的 `0.5|x|erfc` 本身就是小量。

## 预防

- 对输出趋近 0 的尾部，检查公式是否由两个大数相减得到。
- 提高近似函数精度后若误差仍集中在负尾部，优先改公式形态而不是继续堆迭代。
- 用包含较大负输入的 fp32 case 单独观察 MARE/MERE。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：基线 18/20，负尾失败用例 MARE 约 0.855。（来源平台：未定位；证据等级: tool_observed）。
- 修复后：AFTER 20/20，通过率 100%。（来源平台：未定位；证据等级: tool_observed）。

## 失效边界

- `erfc` 近似本身仍须有足够精度；本轨迹用 Reciprocal 加两次 Newton 修正。
- tanh approximate 分支没有使用本改写。
- 其他平台和更极端输入范围需复测。

## 坏实践

- 对负尾部直接计算 `0.5*x + 0.5*|x|*erf`。
- 把所有尾部误差归因于 `Erf` 指令精度，忽略最终加法的条件数。
- 只测 [-1,1]，不覆盖 GELU 的负大值尾部。

## 误诊

这不是 Cast 精度或 padding 污染：失败只集中在 fp32 exact 的负大值尾部，且改公式形态即可让同一 harness 全通过。