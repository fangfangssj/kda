---
type: "Runbook"
title: "A3 RMSNorm 不要用分母正值保护吞掉 NaN 传播"
description: "RMSNorm 的规约含 NaN 时，denom > 0 的倒数保护会把 NaN 分母映射成 0，破坏整行 NaN 传播；在 epsilon 保证有限零输入分母为正的前提下直接求倒数。"
tags: ["task.debugging", "topic.compute.normalization", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# A3 RMSNorm 不要用分母正值保护吞掉 NaN 传播

## 现象

在 A3（Ascend910_9362/dav-2201）的 Ascend C RMSNorm 中，普通有限输入已通过，但官方正确性用例仍报告 NaN 位置不匹配。全 NaN 的手工输入会掩盖问题；把有限值和 NaN 混在同一归约行后，golden 的输出整行都是 NaN，而 kernel 产生了 114287 个有限值。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`RMSNorm`、`FinalizeRstd`、`NaN`、`denom` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

当规约输入含 NaN 时，平方和、均值及开方结果都应继续为 NaN，使同一行的 RMSNorm 输出传播 NaN。但是 `NaN > 0.0f` 为 false，条件表达式把本应为 NaN 的 `rstd` 改成了 `0.0f`；随后 `x * 0 * gamma` 可产生有限零值，因而改变 NaN 位置。这里的保护也不负责正常的除零防护：全零有限输入仍有正的 `sqrt(epsilon)`。

## 解决配方

```cpp
__aicore__ inline float FinalizeRstd(const AscendC::LocalTensor<float>& sumBuf)
{
    constexpr int32_t W = 8;   // 8 fp32 = 32B aligned slot; only [0] is real
    float avgFactor = (numCol_ > 0) ? (1.0f / static_cast<float>(numCol_)) : 0.0f;
    AscendC::Muls(sumBuf, sumBuf, avgFactor, W);     // sumBuf[0] = Σx² · (1/D)
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(sumBuf, sumBuf, epsilon_, W);      // sumBuf[0] = avg + ε
    AscendC::PipeBarrier<PIPE_V>();
    auto sqrtBuf = tmpBuf_.Get<float>();             // reuse tmpBuf_ (ReduceSum done)
    AscendC::Sqrt(sqrtBuf, sumBuf, W);               // sqrtBuf[0] = √(avg+ε)
    AscendC::PipeBarrier<PIPE_V>();
    float denom = sqrtBuf.GetValue(0);
    return (denom > 0.0f) ? (1.0f / denom) : 0.0f;
}
```

```cpp
__aicore__ inline float FinalizeRstd(const AscendC::LocalTensor<float>& sumBuf)
{
    constexpr int32_t W = 8;   // 8 fp32 = 32B aligned slot; only [0] is real
    float avgFactor = (numCol_ > 0) ? (1.0f / static_cast<float>(numCol_)) : 0.0f;
    AscendC::Muls(sumBuf, sumBuf, avgFactor, W);     // sumBuf[0] = Σx² · (1/D)
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(sumBuf, sumBuf, epsilon_, W);      // sumBuf[0] = avg + ε
    AscendC::PipeBarrier<PIPE_V>();
    auto sqrtBuf = tmpBuf_.Get<float>();             // reuse tmpBuf_ (ReduceSum done)
    AscendC::Sqrt(sqrtBuf, sumBuf, W);               // sqrtBuf[0] = √(avg+ε)
    AscendC::PipeBarrier<PIPE_V>();
    // No clamping guard: a NaN reduction must propagate through rstd to the row.
    return 1.0f / sqrtBuf.GetValue(0);
}
```

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：NaN 定向探针输出 nan-pos match: False。（来源平台：a3；证据等级: tool_observed）。
- 修复后：移除正值 guard 后边界与常规用例逐项通过。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

- `reproduce_count=1`，只来自一条真实轨迹。
- 本卡只解释 NaN 位置错误，不把同轨迹中的 gamma 队列依赖、UB 容量或性能调整合并为同一根因。
- 前提是算子语义要求 IEEE NaN 传播，且合法输入保证 `epsilon > 0`；若 API 明确定义 NaN 清洗或允许非正 epsilon，需另行处理。
