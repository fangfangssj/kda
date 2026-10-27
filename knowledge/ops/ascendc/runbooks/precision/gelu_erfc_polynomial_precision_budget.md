---
type: "Runbook"
title: "AscendC GELU 三系数 erfc 在大值域 fp32 精度不足"
description: "GELU exact 路径为减少 Vector 指令把 erfc 改成三系数近似后，两个大值域 fp32 用例超出 MARE 阈值；恢复五系数多项式并保留无相消公式后通过全部用例。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.activation", "topic.compute.elementwise.math_function"]
status: draft
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
---

# AscendC GELU 三系数 erfc 在大值域 fp32 精度不足

## 现象

AscendC GELU 的 exact 路径用 `max(x,0) - 0.5*abs(x)*erfc(abs(x)/sqrt(2))` 避免负大值相消。为减少 Vector 指令，将 erfc 从五系数近似缩成三系数后，评测只通过 `18/20`：`gelu_11` 与 `gelu_20` 的 fp32 `MARE` 分别约 `0.003769`、`0.003770`，超过 `0.001220703` 阈值。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`GELU`、`ComputeExact`、`AscendC::Exp`、`AscendC::Reciprocal` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 未定位；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

三系数 Abramowitz 7.1.25 的绝对近似误差在 `erfc` 极小时被 GELU 尾部的 `0.5*abs(x)` 放大；golden 此时接近零，因而相对误差指标显著超限。失败只集中在两个大值域 fp32 用例，换回误差更小的五系数 7.1.26 后同一套用例全过，故根因不是 tiling、搬运或 dtype Cast，而是三系数近似的精度预算不足。

## 解决配方

### before

```cpp
    __aicore__ inline void ComputeExact(
        const AscendC::LocalTensor<float>& xF,
        const AscendC::LocalTensor<float>& w,
        const AscendC::LocalTensor<float>& t,
        const AscendC::LocalTensor<float>& expBuf,
        int64_t count)
    {
        // y = max(x, 0) - 0.5*|x| * erfc(|x|/sqrt(2))
        // erfc(u) computed via Abramowitz 7.1.25 (3 coeffs, err < 2.5e-5)
        constexpr float kSqrtHalf = 0.7071067811865475f;
        constexpr float kP   = 0.47047f;
        constexpr float kA1  = 0.3480242f;
        constexpr float kA2  = -0.0958798f;
        constexpr float kA3  = 0.7478556f;

        // u = |x|/sqrt(2) into t
        AscendC::Abs(t, xF, count);
        AscendC::Muls(t, t, kSqrtHalf, count);
        // expBuf = exp(-u^2)
        AscendC::Mul(expBuf, t, t, count);
        AscendC::Muls(expBuf, expBuf, -1.0f, count);
        AscendC::Exp(expBuf, expBuf, count);
        // denom = 1+p*u in w, t = 1/denom (refined by Newton)
        AscendC::Muls(w, t, kP, count);
        AscendC::Adds(w, w, 1.0f, count);
        AscendC::Reciprocal(t, w, count);
        AscendC::Mul(w, w, t, count);
        AscendC::Muls(w, w, -1.0f, count);
        AscendC::Adds(w, w, 2.0f, count);
        AscendC::Mul(t, t, w, count);
        // 3-coeff Horner: poly = a3*t^3 + a2*t^2 + a1*t = t*(a1 + t*(a2 + t*a3))
        AscendC::Muls(w, t, kA3, count);
        AscendC::Adds(w, w, kA2, count); AscendC::Mul(w, w, t, count);
        AscendC::Adds(w, w, kA1, count); AscendC::Mul(w, w, t, count);
        // erfc = poly * exp(-u^2)
        AscendC::Mul(w, w, expBuf, count);
        // y = max(x, 0) - 0.5*|x|*erfc
        AscendC::Abs(expBuf, xF, count);
        AscendC::Muls(expBuf, expBuf, 0.5f, count);
        AscendC::Mul(w, w, expBuf, count);              // w = 0.5*|x|*erfc
        AscendC::Maxs(t, xF, 0.0f, count);             // t = max(x, 0)
        AscendC::Sub(w, t, w, count);                   // w = max(x,0) - 0.5*|x|*erfc
    }
```

### after

```cpp
    __aicore__ inline void ComputeExact(
        const AscendC::LocalTensor<float>& xF,
        const AscendC::LocalTensor<float>& w,
        const AscendC::LocalTensor<float>& t,
        const AscendC::LocalTensor<float>& expBuf,
        int64_t count)
    {
        // y = max(x, 0) - 0.5*|x| * erfc(|x|/sqrt(2))
        // erfc(u) via Abramowitz 7.1.26 (5 coeffs, err < 1.5e-7)
        constexpr float kSqrtHalf = 0.7071067811865475f;
        constexpr float kP   = 0.3275911f;
        constexpr float kA0  = 0.254829592f;
        constexpr float kA1  = -0.284496736f;
        constexpr float kA2  = 1.421413741f;
        constexpr float kA3  = -1.453152027f;
        constexpr float kA4  = 1.061405429f;

        AscendC::Abs(t, xF, count);
        AscendC::Muls(t, t, kSqrtHalf, count);
        AscendC::Mul(expBuf, t, t, count);
        AscendC::Muls(expBuf, expBuf, -1.0f, count);
        AscendC::Exp(expBuf, expBuf, count);
        AscendC::Muls(w, t, kP, count);
        AscendC::Adds(w, w, 1.0f, count);
        AscendC::Reciprocal(t, w, count);
        AscendC::Mul(w, w, t, count);
        AscendC::Muls(w, w, -1.0f, count);
        AscendC::Adds(w, w, 2.0f, count);
        AscendC::Mul(t, t, w, count);
        AscendC::Muls(w, t, kA4, count);
        AscendC::Adds(w, w, kA3, count); AscendC::Mul(w, w, t, count);
        AscendC::Adds(w, w, kA2, count); AscendC::Mul(w, w, t, count);
        AscendC::Adds(w, w, kA1, count); AscendC::Mul(w, w, t, count);
        AscendC::Adds(w, w, kA0, count); AscendC::Mul(w, w, t, count);
        AscendC::Mul(w, w, expBuf, count);    // w = erfc(|x|/sqrt(2))
        // y = max(x, 0) - 0.5*|x|*erfc
        AscendC::Abs(expBuf, xF, count);
        AscendC::Muls(expBuf, expBuf, 0.5f, count);
        AscendC::Mul(w, w, expBuf, count);
        AscendC::Maxs(t, xF, 0.0f, count);
        AscendC::Sub(w, t, w, count);
    }
```

## 设计决策

保留无相消的 GELU 公式和一次 Newton 修正，只恢复 erfc 多项式的精度阶数。这样故障变量只有近似系数族，不把修复与 tile、队列或 Cast 路线混合。

## 性能影响

失败的三系数版本在通过的 18 个用例上；五系数修复版，说明精度恢复有额外 Vector 指令代价。本卡只主张精度闭环，不主张性能提升。

## 预防

- 近似函数的误差界必须换算到最终算子输出及实际误差指标，不能只比较近似函数的绝对误差；
- GELU exact 路径至少保留大正/负值域 fp32 用例；
- 降阶优化后必须重新跑全部精度用例，不能只看通过用例的性能分。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：基线仅 18/20，通过失败用例的 MARE 分别为 0.003769 与 0.003770。（来源平台：未定位；证据等级: tool_observed）。
- 修复后：AFTER 20/20，通过率 100%。（来源平台：未定位；证据等级: tool_observed）。

## 失效边界

本结论不直接覆盖其他平台、其他 erfc 系数族、分段近似或不同精度阈值。若故障随 tile、地址或 dtype 随机变化，应先排查搬运、同步和 Cast，而非套用本修复。

## 坏实践

- 用“近似误差小于 fp32 MERE 阈值”直接推出最终 GELU 的 MARE 必然通过；
- 只观察平均性能，将少量大值域精度失败当作可忽略异常；
- 同时修改公式、tiling 和系数，导致无法归因。

## 误诊

该故障稳定集中在大值域 fp32 用例，输出误差由多项式精度阶数决定。若所有 dtype、所有值域都失败，先检查公式选择、approximate 路由和参数绑定。