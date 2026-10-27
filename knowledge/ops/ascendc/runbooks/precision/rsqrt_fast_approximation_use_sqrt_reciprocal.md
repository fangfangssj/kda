---
type: "Runbook"
title: "Rsqrt 精度差异须区分 Sqrt 后取倒数路径"
description: "保留a3的Sqrt加标量倒数修复；新增a2/CANN9.1原生三路径探针，向量Div与Reciprocal误差不可混同。"
tags: ["task.optimization", "topic.compute.normalization", "topic.compute.elementwise.math_function", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: '2026-10-20T12:33:33Z'
platforms: ["a3", "a2"]
aliases: ["KernelRmsNorm::Compute"]
---

# Rsqrt 精度差异须区分 Sqrt 后取倒数路径
## 平台分支与路线选择

原有a3/CANN unknown的RmsNorm问题使用 `Sqrt` 后标量 `1.0f/s`，其原验证信号完整保留。新增a2/CANN9.1.0仅校准原生Rsqrt、Sqrt+向量Div和Sqrt+向量Reciprocal；不能以新向量Div的结果认证旧标量路线，也不把新设备当旧实验身份。两分支共用固定inv_std输入的诊断方法，具体已验证路线分别限定。


## 现象

RmsNorm 的 fp32 用例出现约 `1e-2` 最大输出差，而 fp16/bf16 因容差较宽可能通过。确定性 `x=1, gamma=1, eps=1e-6` 探针跨多个短、长归约长度都给出 `rstd=0.998046875`，期望约 `0.9999995`；随机输入也出现 `1.03515625` 对 `1.03701699` 的稳定偏差。整齐的二进制量化值是快速倒数精度不足的判别信号。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`AscendC::Rsqrt`、`AscendC::Sqrt`、`dav-2201`、`rstd` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

原记录中，不带 config 的 `Rsqrt<float>` 调用在 dav-2201 路径上产生约 9-bit 量级的快速倒数结果。规约、均值和加 epsilon 链条随 D 正常变化，固定 ones 输入对所测 D 得到相同低精度 rstd。这些观测支持在该轨迹的 API 调用路径上定位精度差异；未对最终生成的单条硬件指令、SDK 内部精化实现或全部底层原因作独立鉴定。

## 解决配方

### before

```cpp
    __aicore__ inline void Compute(int64_t r)
    {
        (void)r;
        auto scalar = scalarBuf_.Get<float>();
        auto sq = sqBuf_.Get<float>();
        auto dummy = dummyBuf_.Get<float>();

        if constexpr (std::is_same<T, float>::value) {
            auto xFp32 = inQueueX_.DeQue<float>();
            AscendC::Mul(sq, xFp32, xFp32, numCol_);                              // x^2
            AscendC::ReduceSum<float>(scalar, sq, dummy, numCol_);               // sum(x^2)
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(scalar, scalar, 1.0f / static_cast<float>(numCol_), 1); // mean
            AscendC::Adds(scalar, scalar, epsilon_, 1);                         // mean + eps
            AscendC::Rsqrt<float>(scalar, scalar, 1);                           // rstd = 1/sqrt(mean+eps)
            AscendC::PipeBarrier<PIPE_V>();
            float rstd = scalar.GetValue(0);

            auto yOut = outQueueY_.AllocTensor<float>();
            AscendC::Muls(yOut, xFp32, rstd, numCol_);                          // x * rstd
            AscendC::Mul(yOut, yOut, gammaFp32Local_, numCol_);                 // * gamma
            outQueueY_.EnQue(yOut);
            inQueueX_.FreeTensor(xFp32);
        } else {
            auto xT = inQueueX_.DeQue<T>();
            auto xFp32 = xFp32Buf_.Get<float>();
            AscendC::Cast(xFp32, xT, AscendC::RoundMode::CAST_NONE, numCol_);  // x -> fp32
            inQueueX_.FreeTensor(xT);
            AscendC::Mul(sq, xFp32, xFp32, numCol_);                            // x^2
            AscendC::ReduceSum<float>(scalar, sq, dummy, numCol_);              // sum(x^2)
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(scalar, scalar, 1.0f / static_cast<float>(numCol_), 1); // mean
            AscendC::Adds(scalar, scalar, epsilon_, 1);                        // mean + eps
            AscendC::Rsqrt<float>(scalar, scalar, 1);                          // rstd = 1/sqrt(mean+eps)
            AscendC::PipeBarrier<PIPE_V>();
            float rstd = scalar.GetValue(0);

            AscendC::Muls(sq, xFp32, rstd, numCol_);                           // y = x * rstd (reuse sqBuf)
            AscendC::Mul(sq, sq, gammaFp32Buf_.Get<float>(), numCol_);         // * gamma
            auto yOutT = outQueueY_.AllocTensor<T>();
            constexpr auto roundMode = std::is_same<T, bfloat16_t>::value
                ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(yOutT, sq, roundMode, numCol_);                      // y -> T
            outQueueY_.EnQue(yOutT);
        }
    }
```

### after

```cpp
    __aicore__ inline void Compute(int64_t r)
    {
        (void)r;
        auto scalar = scalarBuf_.Get<float>();
        auto sq = sqBuf_.Get<float>();
        auto dummy = dummyBuf_.Get<float>();

        if constexpr (std::is_same<T, float>::value) {
            auto xFp32 = inQueueX_.DeQue<float>();
            AscendC::Mul(sq, xFp32, xFp32, numCol_);                              // x^2
            AscendC::ReduceSum<float>(scalar, sq, dummy, numCol_);               // sum(x^2)
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(scalar, scalar, 1.0f / static_cast<float>(numCol_), 1); // mean
            AscendC::Adds(scalar, scalar, epsilon_, 1);                         // mean + eps
            AscendC::Sqrt(scalar, scalar, 1);                                  // sqrt(mean+eps)
            AscendC::PipeBarrier<PIPE_V>();
            float s = scalar.GetValue(0);
            float rstd = 1.0f / s;  // rstd = 1/sqrt(mean+eps); NaN/inf propagate (mean+eps >= eps > 0 for finite)

            auto yOut = outQueueY_.AllocTensor<float>();
            AscendC::Muls(yOut, xFp32, rstd, numCol_);                          // x * rstd
            AscendC::Mul(yOut, yOut, gammaFp32Local_, numCol_);                 // * gamma
            outQueueY_.EnQue(yOut);
            inQueueX_.FreeTensor(xFp32);
        } else {
            auto xT = inQueueX_.DeQue<T>();
            auto xFp32 = xFp32Buf_.Get<float>();
            AscendC::Cast(xFp32, xT, AscendC::RoundMode::CAST_NONE, numCol_);  // x -> fp32
            inQueueX_.FreeTensor(xT);
            AscendC::Mul(sq, xFp32, xFp32, numCol_);                            // x^2
            AscendC::ReduceSum<float>(scalar, sq, dummy, numCol_);              // sum(x^2)
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(scalar, scalar, 1.0f / static_cast<float>(numCol_), 1); // mean
            AscendC::Adds(scalar, scalar, epsilon_, 1);                        // mean + eps
            AscendC::Sqrt(scalar, scalar, 1);                                 // sqrt(mean+eps)
            AscendC::PipeBarrier<PIPE_V>();
            float s = scalar.GetValue(0);
            float rstd = 1.0f / s;  // rstd = 1/sqrt(mean+eps); NaN/inf propagate (mean+eps >= eps > 0 for finite)

            AscendC::Muls(sq, xFp32, rstd, numCol_);                           // y = x * rstd (reuse sqBuf)
            AscendC::Mul(sq, sq, gammaFp32Buf_.Get<float>(), numCol_);         // * gamma
            auto yOutT = outQueueY_.AllocTensor<T>();
            constexpr auto roundMode = std::is_same<T, bfloat16_t>::value
                ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(yOutT, sq, roundMode, numCol_);                      // y -> T
            outQueueY_.EnQue(yOutT);
        }
    }
```

## 设计决策

保留 fp32 规约与既有缓冲，只替换 rstd 的最后一步，避免把精度问题误治为 tiling 或 Cast 问题。有限输入下 `mean+eps > 0`，无需 `s>0` 保护；该保护会在 `s=NaN` 时错误选择 0，破坏整行 NaN 传播。

## 性能影响

新路线增加一次标量读回和标量除法，轨迹未提供配对性能数据；本卡只主张精度恢复，不主张性能收益。

## 预防

- 对快速倒数/倒数平方根 API 用 `x=1` 和非整平方输入做目标平台精度探针；
- 分 dtype 检查，避免低精度宽容差掩盖 fp32 的系统偏差；
- 不把其他平台或带 config 形态的精度承诺迁移到当前调用形态。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：default-eps fp32 有限值最大差 1.315689e-02，边界子集汇总 5 failed、5 passed。（来源平台：a3；证据等级: tool_observed）。
- 修复后：Sqrt+标量倒数实现及 NaN 传播修正后完整 NPU 测试 167 passed。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

上述原有修复结论只在 a3 路径验证，不覆盖 a2、950、带精度 config 的 API 或其他 CANN 版本。下文新增 a2 仅为向量接口校准，未复验该标量修复；扩展修法前须按目标设备精确判型并复测。

## 坏实践

- 看到 `Rsqrt<float>` 就假设结果具有完整 fp32 精度；
- 只跑 fp16/bf16 用例，用宽容差宣布数值路径正确；
- 用 `s>0 ? 1/s : 0` 保护有限域，却忽略 NaN 比较恒为 false 会破坏传播。

## 误诊

若 rstd 误差随 D、行号或输入地址随机变化，先查规约、同步和索引；本问题的特征是同一数学输入得到稳定、整齐的低精度量化值。

## A2/CANN9.1 独立新增探针

### 现象

在排查归一化参考的inv_std差异时，单独固定进入倒数平方根步骤的FP32输入，可以把归约误差与最后数学接口的误差分开。本次直接对同一256元素张量运行三条AscendC路径：不带显式config的Rsqrt、Sqrt后向量Div(1,sqrt(x))、Sqrt后向量Reciprocal，并调用实际torch_npu的torch.rsqrt对照。

三个原生路径并非等价精度。相对于CPU binary64 sqrt/divide后cast成binary32的oracle，在251个参考有限且非零输出上，最大相对误差分别约2.869e-3、5.960e-8、1.953e-3。框架rsqrt与Sqrt+Div在全部256个输出的原始bits相同；Sqrt+Reciprocal没有获得Div的精度。

**与 [OL-103](ol_103_npu_transcendentals_fp16_precision.md) 的测量对象边界。** OL-103 记录了 single_run 的硬件近似精度分层。本节测量的是上述 CANN 9.1.0、Ascend910B4、精确接口路径和固定输入上的组合 API 最终输出；未隔离 Sqrt / Div 单条硬件指令，也未鉴定编译器 lowering 或 SDK 内部精化实现。因此，本节不是对 OL-103 原实验或其单指令解释的同条件复验。

采用 OL-103 的分层时，需要核对 CANN 版本、精确重载 / config、输入和参考计算。该卡没有给出与本探针逐项匹配的这些条件，不能仅因操作名相同，就把其 Tier 2 概括覆盖本节 SDK 组合的历史探针结果。本节误差仍限于 251 个有限非零参考项，框架 raw bits 对照限于所列全部 256 项；其他版本、重载、配置、输入域和完整归一化结果需要各自验收。

### 适用场景

- 工程形态：AscendC AIV_ONLY直调，固定归约之后的实际输入进行接口校准。
- API路径：`Rsqrt(y,x,n)`、`Sqrt(t,x,n); Div(y,ones,t,n)`、`Sqrt(t,x,n); Reciprocal(y,t,n)`，n=256，float32。
- 验证平台与版本：a2/Ascend910B4，CANN9.1.0，torch/torch_npu2.9.0。
- 适用粒度：`exact_task`，只覆盖这些精确重载/配置和输入。
- 路线优先级：`recommended_fastpath`，指诊断路径；未做正确实现之间的性能排序。
- 未验证：旧LayerNorm约1e-2历史主张、scalar 1.0f/sqrt(var)、其他config/dtype/版本，以及完整归一化最终输出与成本。

### 判别方法

先冻结真正传入此步骤的var+eps或mean_square+eps，不把“参考张量是FP32”当内部接口证明。本次输入为240个`float32(2^((i-120)/2))`加16个明确bits的控制，包含最小subnormal、最大subnormal、最小normal、最大finite、零/负零、负数、Inf、NaN及1附近。比较误差定义为`abs((y - float32(1/sqrt_float64(x)))/oracle)`；只有oracle有限且非零的251项进入分母，特殊值单独保存。

以下为实际三路径微核完整函数，逐字恢复：

```cpp
#include "kernel_operator.h"

template <typename T>
__aicore__ inline void CompareTruthTable(GM_ADDR a, GM_ADDR b, GM_ADDR out)
{
    constexpr uint32_t count = 256;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qa, qb;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qo;
    pipe.InitBuffer(qa, 1, count * sizeof(T));
    pipe.InitBuffer(qb, 1, count * sizeof(T));
    pipe.InitBuffer(qo, 1, count / 8);
    AscendC::GlobalTensor<T> ga, gb;
    AscendC::GlobalTensor<uint8_t> go;
    ga.SetGlobalBuffer((__gm__ T *)a, count);
    gb.SetGlobalBuffer((__gm__ T *)b, count);
    go.SetGlobalBuffer((__gm__ uint8_t *)out, 6 * count / 8);
    auto la = qa.AllocTensor<T>();
    auto lb = qb.AllocTensor<T>();
    AscendC::DataCopy(la, ga, count);
    AscendC::DataCopy(lb, gb, count);
    qa.EnQue(la);
    qb.EnQue(lb);
    la = qa.DeQue<T>();
    lb = qb.DeQue<T>();
    const AscendC::CMPMODE modes[6] = {
        AscendC::CMPMODE::EQ, AscendC::CMPMODE::NE,
        AscendC::CMPMODE::LT, AscendC::CMPMODE::LE,
        AscendC::CMPMODE::GT, AscendC::CMPMODE::GE};
    for (uint32_t m = 0; m < 6; ++m) {
        auto lo = qo.AllocTensor<uint8_t>();
        AscendC::Compare(lo, la, lb, modes[m], count);
        qo.EnQue(lo);
        lo = qo.DeQue<uint8_t>();
        AscendC::DataCopy(go[m * count / 8], lo, count / 8);
        qo.FreeTensor(lo);
    }
    qa.FreeTensor(la);
    qb.FreeTensor(lb);
}

extern "C" __global__ __aicore__ void compare_float(GM_ADDR a, GM_ADDR b, GM_ADDR out)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CompareTruthTable<float>(a, b, out);
}
extern "C" __global__ __aicore__ void compare_half(GM_ADDR a, GM_ADDR b, GM_ADDR out)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CompareTruthTable<half>(a, b, out);
}
extern "C" __global__ __aicore__ void rsqrt_paths(GM_ADDR input, GM_ADDR output, uint32_t mode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t n = 256;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qi;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> temp, ones;
    pipe.InitBuffer(qi, 1, n * sizeof(float));
    pipe.InitBuffer(qo, 1, n * sizeof(float));
    pipe.InitBuffer(temp, n * sizeof(float));
    pipe.InitBuffer(ones, n * sizeof(float));
    AscendC::GlobalTensor<float> gi, go;
    gi.SetGlobalBuffer((__gm__ float *)input, n);
    go.SetGlobalBuffer((__gm__ float *)output, n);
    auto x = qi.AllocTensor<float>();
    AscendC::DataCopy(x, gi, n);
    qi.EnQue(x);
    x = qi.DeQue<float>();
    auto y = qo.AllocTensor<float>();
    auto t = temp.Get<float>();
    auto one = ones.Get<float>();
    if (mode == 0) {
        AscendC::Rsqrt(y, x, n);
    } else {
        AscendC::Sqrt(t, x, n);
        if (mode == 1) {
            AscendC::Duplicate(one, 1.0f, n);
            AscendC::Div(y, one, t, n);
        } else {
            AscendC::Reciprocal(y, t, n);
        }
    }
    qo.EnQue(y);
    qi.FreeTensor(x);
    y = qo.DeQue<float>();
    AscendC::DataCopy(go, y, n);
    qo.FreeTensor(y);
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void compare_float_do(uint32_t n, void *s, uint8_t *a, uint8_t *b, uint8_t *out)
{
    compare_float<<<n, nullptr, s>>>(a, b, out);
}
extern "C" void compare_half_do(uint32_t n, void *s, uint8_t *a, uint8_t *b, uint8_t *out)
{
    compare_half<<<n, nullptr, s>>>(a, b, out);
}
extern "C" void rsqrt_paths_do(uint32_t n, void *s, uint8_t *input, uint8_t *output, uint32_t mode)
{
    rsqrt_paths<<<n, nullptr, s>>>(input, output, mode);
}
#endif
```

宿主检查与launch的实际完整代码如下；模式值只选择上述三条分支。

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
extern "C" void compare_float_do(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void compare_half_do(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void rsqrt_paths_do(uint32_t, void *, uint8_t *, uint8_t *, uint32_t);
static uint8_t *ptr(const at::Tensor &t)
{
    return static_cast<uint8_t *>(const_cast<void *>(t.data_ptr()));
}
at::Tensor compare_modes(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.numel() == 256 && b.numel() == 256, "probe needs 256 elements");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "contiguous inputs required");
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.device() == b.device(), "matching inputs required");
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
    auto out = at::empty({6, 32}, a.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    if (a.scalar_type() == at::kFloat) compare_float_do(1, stream, ptr(a), ptr(b), ptr(out));
    else {
        TORCH_CHECK(a.scalar_type() == at::kHalf, "float32/float16 only");
        compare_half_do(1, stream, ptr(a), ptr(b), ptr(out));
    }
    return out;
}
at::Tensor inverse_sqrt(const at::Tensor &a, uint32_t mode)
{
    TORCH_CHECK(a.numel() == 256 && a.is_contiguous() && a.scalar_type() == at::kFloat, "256 contiguous float32 elements");
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1 && mode <= 2, "NPU input and mode 0..2 required");
    auto out = at::empty_like(a);
    rsqrt_paths_do(1, c10_npu::getCurrentNPUStream().stream(false), ptr(a), ptr(out), mode);
    return out;
}
PYBIND11_MODULE(_kb_compare_nan_probe, m)
{
    m.def("compare_modes", &compare_modes);
    m.def("inverse_sqrt", &inverse_sqrt);
}
```

实际命令为`python3 build_ascendc.py . --soc-version Ascend910B4`与`python3 run_probe.py`；完整driver及256项输入/四条输出的raw bits在公开可审材料保留。不能把这里的Sqrt+Div称为旧源码的Sqrt+scalar divide。

| 路径 | 有效误差分母 | 最大相对误差 |
|---|---|---|
|AscendC_Rsqrt|251|0.0028689360272363055|
|AscendC_Sqrt_Div|251|5.9604648328104516e-08|
|AscendC_Sqrt_Reciprocal|251|0.001953125|
|torch_npu_rsqrt|251|5.9604648328104516e-08|

例如x=3，Rsqrt为0.576171875，Sqrt+Div与框架均为0.5773502588272095，Sqrt+Reciprocal为0.578125。x=0.5时三原生值依次为1.41015625、1.4142135381698608、1.4140625。差异来自此次实际路径输出，不是对所有同名API的精度承诺。

#### 本轮原生探针的完整输入、解码与误差统计 driver

以下为实际运行的 run_probe.py 全文。本文件同时运行 Compare 与倒数平方根两个探针；本页只采用对应分支的观测，另一分支保留为同文件上下文，不增加本页结论范围。

```python
"""Measure the actual Ascend C Compare overload, preserving input and mask bits."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch_npu

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "kernel/build"))
import _kb_compare_nan_probe as ext


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.npu.set_device(0)
    rows = []
    tensors = {}
    modes = ["EQ", "NE", "LT", "LE", "GT", "GE"]
    bits_by_dtype = [
        ("float32", np.uint32, np.float32,
         [0x00000000, 0x80000000, 0x3f800000, 0xc0000000, 0x7f800000,
          0xff800000, 0x7fc00001, 0xffc00001, 0x7f800001]),
        ("float16", np.uint16, np.float16,
         [0x0000, 0x8000, 0x3c00, 0xc000, 0x7c00, 0xfc00, 0x7e01, 0xfe01, 0x7c01]),
    ]
    for dtype, uint, floating, patterns in bits_by_dtype:
        labels = ["+0", "-0", "+1", "-2", "+inf", "-inf", "+qnan", "-qnan", "+snan"]
        pairs = [(i, j) for i in range(len(patterns)) for j in range(len(patterns))]
        pairs = (pairs * 4)[:256]
        ba = np.asarray([patterns[i] for i, _ in pairs], dtype=uint)
        bb = np.asarray([patterns[j] for _, j in pairs], dtype=uint)
        a = torch.from_numpy(ba.view(floating).copy())
        b = torch.from_numpy(bb.view(floating).copy())
        an, bn = a.to("npu:0"), b.to("npu:0")
        torch.npu.synchronize()
        a_roundtrip = an.cpu().numpy().view(uint)
        b_roundtrip = bn.cpu().numpy().view(uint)
        packed = ext.compare_modes(an, bn)
        torch.npu.synchronize()
        cpu_packed = packed.cpu()
        unpacked = np.unpackbits(cpu_packed.numpy(), axis=1, bitorder="little")
        with np.errstate(invalid="ignore"):
            av, bv = ba.view(floating), bb.view(floating)
            expected = np.stack([av == bv, av != bv, av < bv, av <= bv, av > bv, av >= bv])
        nan = np.isnan(av) | np.isnan(bv)
        finite = np.isfinite(av) & np.isfinite(bv)
        observed_cases = []
        for k, (i, j) in enumerate(pairs[:81]):
            observed_cases.append({
                "a_label": labels[i], "b_label": labels[j],
                "a_bits": hex(int(ba[k])), "b_bits": hex(int(bb[k])),
                "actual": {m: int(unpacked[n, k]) for n, m in enumerate(modes)},
                "cpu_ieee_oracle": {m: int(expected[n, k]) for n, m in enumerate(modes)},
            })
        rows.append({
            "dtype": dtype, "elements": 256, "mode_order": modes,
            "input_a_sha256": hashlib.sha256(ba.tobytes()).hexdigest(),
            "input_b_sha256": hashlib.sha256(bb.tobytes()).hexdigest(),
            "roundtrip_a_bits_equal": bool(np.array_equal(ba, a_roundtrip)),
            "roundtrip_b_bits_equal": bool(np.array_equal(bb, b_roundtrip)),
            "packed_output_hex": cpu_packed.numpy().tobytes().hex(),
            "finite_mismatches_per_mode": {m: int(np.sum(unpacked[n, finite] != expected[n, finite])) for n, m in enumerate(modes)},
            "nan_pairs": int(nan.sum()),
            "nan_pair_true_counts": {m: int(unpacked[n, nan].sum()) for n, m in enumerate(modes)},
            "cpu_ieee_mismatches_per_mode": {m: int(np.sum(unpacked[n] != expected[n])) for n, m in enumerate(modes)},
            "unique_cases": observed_cases,
        })
        tensors[dtype] = {"a_bits": torch.from_numpy(ba.copy()), "b_bits": torch.from_numpy(bb.copy()), "mask": cpu_packed}
    # Separate C066 experiment: basic Ascend C Rsqrt versus two explicit paths.
    positive = np.asarray([np.float32(2.0 ** ((i - 120) / 2)) for i in range(240)], dtype=np.float32)
    special_bits = np.asarray([0, 0x80000000, 1, 0x007fffff, 0x00800000, 0xbf800000,
                               0x7f800000, 0x7fc00001, 0x3f800001, 0x3f7fffff,
                               0x7f7fffff, 0x3f000000, 0x40000000, 0x40400000,
                               0x3dcccccd, 0x41200000], dtype=np.uint32)
    rx = torch.from_numpy(np.concatenate([positive, special_bits.view(np.float32)]))
    rn = rx.to("npu:0")
    root_outputs = []
    for m in range(3):
        y = ext.inverse_sqrt(rn, m)
        torch.npu.synchronize()
        root_outputs.append(y.cpu())
    framework = torch.rsqrt(rn)
    torch.npu.synchronize()
    root_outputs.append(framework.cpu())
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        oracle = np.asarray(1.0 / np.sqrt(rx.numpy().astype(np.float64)), dtype=np.float32)
    valid = np.isfinite(oracle) & (oracle != 0)
    rsqrt = {"input_bits_hex": rx.numpy().view(np.uint32).tobytes().hex(),
             "input_sha256": hashlib.sha256(rx.numpy().tobytes()).hexdigest(),
             "oracle": "CPU binary64 sqrt/divide, cast to binary32; not legacy LayerNorm inv_std",
             "mode_order": ["AscendC_Rsqrt", "AscendC_Sqrt_Div", "AscendC_Sqrt_Reciprocal", "torch_npu_rsqrt"],
             "observations": []}
    for name, out in zip(rsqrt["mode_order"], root_outputs):
        y = out.numpy()
        rsqrt["observations"].append({"path": name,
            "output_bits_hex": y.view(np.uint32).tobytes().hex(),
            "reference_finite_nonzero_count": int(valid.sum()),
            "max_relative_error_on_reference_finite_nonzero": float(np.max(np.abs((y[valid].astype(np.float64) - oracle[valid]) / oracle[valid]))),
            "exact_output_bits_vs_binary32_oracle": int(np.sum(y.view(np.uint32) == oracle.view(np.uint32))),
            "nan_count": int(np.isnan(y).sum()), "inf_count": int(np.isinf(y).sum())})
        tensors[name] = out
    tensors["rsqrt_input"] = rx
    tensors["rsqrt_oracle"] = torch.from_numpy(oracle.copy())
    torch.save(tensors, ROOT / "input_output_bits.pt")
    header_roots = [Path("/usr/local/Ascend/cann-9.1.0/include"), Path("/usr/local/Ascend/cann-9.1.0/ascendc/include")]
    headers = []
    for base in header_roots:
        for path in base.glob("**/kernel_operator_vec_cmpsel_intf.h"):
            headers.append({"path": str(path), "realpath": str(path.resolve()), "sha256": digest(path)})
    result = {
        "schema_version": "compare_nan_new_probe.v1", "observed_at_unix": time.time(),
        "device_logical": 0, "device_name": torch.npu.get_device_name(0),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__, "python": sys.version,
        "platform": platform.platform(), "cann_latest_realpath": str(Path("/usr/local/Ascend/ascend-toolkit/latest").resolve()),
        "module_sha256": digest(ext.__file__), "module_path": ext.__file__,
        "script_sha256": digest(__file__), "headers": headers,
        "source_sha256": {p.name: digest(p) for p in sorted((ROOT / "kernel").glob("*.cpp"))},
        "tensor_archive_sha256": digest(ROOT / "input_output_bits.pt"),
        "loaded_ascend_libraries": [v.rstrip() for v in Path("/proc/self/maps").read_text().splitlines() if "/Ascend/" in v],
        "environment": {k: os.environ.get(k) for k in ["ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "ASCEND_INSTALL_PATH"]},
        "historical_session_recovered": False, "rows": rows, "C066_native": rsqrt,
    }
    (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(ROOT / "result.json"), "device": result["device_name"],
                      "summary": [{k: v for k, v in r.items() if k in ["dtype", "finite_mismatches_per_mode", "nan_pair_true_counts", "roundtrip_a_bits_equal", "roundtrip_b_bits_equal"]} for r in rows]}))


if __name__ == "__main__":
    main()
```

### 设计决策

把inv_std唯一输入冻结后逐值比较，避免同时改变scratch、归约和最终Cast而误归因。特别保留向量Div与向量Reciprocal两个不同候选；只写“Sqrt后取倒数”会把本次观测中不同精度的路径混为一谈。没有重跑旧模型，也未将本次精度校准当作旧问题已修复。

### 系统信息

目标设备是Ascend910B4（a2），CANN9.1.0，torch2.9.0+cpu和torch_npu2.9.0。模块通过现有`build_ascendc.py`构建为AIV_ONLY直调，host实际加载模块SHA为7bdf5104b977b07cdd8591a11c25a08c89afd144dee4dd1821de49dc64b7d875。相关文件为`probe.cpp`、`pybind11.cpp`、`run_probe.py`与构建脚本；没有改旧算子、参考、容差或测试。实际自动同步选项未单独显式设置，其边界以完整构建脚本/日志为准，不能外推其他编译配置。

原始结果的`headers`列表因首次metadata查找路径错误而为空，原文件保持原样。随后从真实编译`.o.d`依赖取出头文件路径，并读取同容器原件核对：`aarch64-linux/asc/include/basic_api/kernel_operator_vec_cmpsel_intf.h` SHA3445867ae1ef92a1143cead32a0d727858e1037b501672a86ae37747c00343bf；`kernel_operator_vec_unary_intf.h` SHAed7f6e44c18d15f7eed22eede71ba2cda92f1e3e50b4d0b286adb4ee107987be。该补证没有回写原始result。

### 验证信号

本次实际原生微核输出和框架对照经过独立raw bits解码与误差分母复算。

- AscendC_Rsqrt：251个oracle有限非零项，最大相对误差0.0028689360272363055。（来源平台：a2；证据等级：tool_observed）

- AscendC_Sqrt_Div：251个oracle有限非零项，最大相对误差5.9604648328104516e-08。（来源平台：a2；证据等级：tool_observed）

- AscendC_Sqrt_Reciprocal：251个oracle有限非零项，最大相对误差0.001953125。（来源平台：a2；证据等级：tool_observed）

- torch_npu_rsqrt：251个oracle有限非零项，最大相对误差5.9604648328104516e-08。（来源平台：a2；证据等级：tool_observed）

- 框架rsqrt与Sqrt+Div在全部256个输出raw bits相同，输入hash重算一致；各路径的零、负零、负数、Inf及NaN输出均独立保留，没有把特殊值计成零误差。（来源平台：a2；证据等级：tool_observed）

### 失效边界

这是新A2/CANN9.1.0接口校准，未恢复旧约1e-2原始探针，也未证明旧LayerNorm候选的根因只在Rsqrt。旧a3 Runbook使用Sqrt后标量除法且CANN版本未知，不能替换它原有信号或直接扩大其结论。没有测高精度config、Newton修正、最终Cast或端到端耗时；用于真实修复时仍需同输入/同契约验收。

### 坏实践

从Rsqrt数学定义推导完整FP32精度；把向量Reciprocal与向量Div或标量除法当作同一路径；同时改变多个阶段后把全任务通过归功于一个接口。

### 性能影响

未测性能，不声称Sqrt+Div更快，也不推荐无条件替换所有Rsqrt调用。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/evidence/result.json#/C066_native`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/evidence/result.json#/projection`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/kernel/probe.cpp`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/kernel/pybind11.cpp`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/run_probe.py`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/build_ascendc.py`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。 公开dynamic投影保留数值、设备名、源码/模块或脚本身份及9.1.0版本标记；不包含完整同PID runtime库路径/hash，不能据此独立认证全部加载库。
