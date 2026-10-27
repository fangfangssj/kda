---
type: Runbook
title: 直接 DFT 的向量化轴、FP32 入口与缓冲预算
description: 从两份完整实现比较按输入位置归约与按频率 Lane 累加的依赖，并核对真实 FP32 入口、对齐长度和 InitBuffer 请求边界。
tags: [task.design, task.implementation, topic.compute.reduction, topic.memory, topic.precision]
status: draft
sources: []
created_at: "2026-10-18T15:30:58Z"
updated_at: '2026-10-22T13:17:58Z'
platforms: []
aliases: [HyenaFftSizePaddingRfft, RFFT, DFT]
---

# 直接 DFT 的向量化轴、FP32 入口与缓冲预算

## 适用场景与证据范围

审阅直接 DFT 的改写时，先标出向量 Lane 对应的轴、沿哪个轴保留累计依赖，再按实际入口的 dtype 核算缓冲。本页人工蒸馏了同一 `HyenaFftSizePaddingRfft` 任务的两份 Level2 源码快照，分别记为 source80 和 source96；名称只标识原知识记录索引。下文内联的是原件中的完整代码单位，供静态审阅。

本页保持 `draft`、`platforms: []`：公开附件提供静态核验入口，不是产品支持或执行证明。原件没有登记为本知识仓的固定提交来源根，也没有覆盖整页的产品支持依据。未声明 CANN、编译器或产品版本；没有执行编译、NPU、计时或 SDK 实现复核，没有确认已加载二进制与源码一致。外部依赖的声明及实现仍需在目标环境核对。

公开静态证据已包含在[原贡献审阅包](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)（ZIP SHA-256 `15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`），解包后的精确根目录是 `cannbot_non_level4_split_review_20261019/evidence_public/c062/`，不是本知识仓的目录。该目录共 17 个文件，`manifest.json` 列出 16 个非自身文件及两套原件中 12 个源码文件的 SHA-256，原件代码字节未修改；原始到安全路径的映射仅保留在本地审计。历史验证 JSON 只发布字段投影，明确省略原路径与日志字段，不把改写指针称为原件。

CPU 模型源码是同目录 `static_request_model.py`（SHA-256 `c87ccd8f7e88010e9702d42ee02a71dd91ed929dd00329fc0117b252d8fedc20`）；实际整数尺寸／分配请求输出是 `static_request_model_output.json`（SHA-256 `4e7a03d3809e3abfce5bee7352135aa1d7f0d8322bdc7d97aeda32d5624d680c`）的 `/rows/0` 至 `/rows/5`，对应下文六行表。完整两套源文件在 `source80/`、`source96/` 下，kernel 分别为 `kernel/hyena_rfft_kernel.h`；所有文件身份可逐项按 `manifest.json` 的 `/packet_files`、`/source_files` 核对。模型只验证整数请求账，不是编译、设备容量、精度排序或性能证据。

## 先确认源码可达的 FP32 入口

两份 `ModelNew.forward` 都执行 `x.float()`，再调用 `_hyena_rfft_ext.run_hyena_rfft`。source80 的 pybind 同时声明三种 dtype 路由，但从该 wrapper 到达的是 `hyena_rfft_do_fp32`；source96 pybind 只接受 FP32，并固定该 launcher。两者均要求三维 contiguous 张量；`x.float()` 本身不能代替连续性检查。输出均分配为 FP32 的 `[B, C, L+1]`。

source80 传给 launcher 的输入地址是 `x.storage().data()`，即 storage 基址，代码没有补上张量的非零 `storage_offset`。contiguous 只约束布局，不能据此断言 storage offset 为零；`x.float()` 对已经是 FP32 的输入也不能作为必然复制到新 storage 的保证。因此即使输入通过 contiguous 检查，带非零 storage offset 的 FP32 view 仍可能从错误起点读取；审阅 source80 的这条原始调用链时，应额外要求转换后输入的 storage offset 为零。source96 使用 `x.data_ptr()` 传张量数据指针，与 source80 的 storage 基址取址不同。本项是 API 取址与源码检查条件的静态风险分析，没有进行 Torch 或设备实验。

pybind 设置 `N=2*L`、`totalSeq=B*C`，以 `totalSeq` 作为 blockDim。launcher 构造 `HyenaRfftKernel<float>`，依次调用 `Init` 和 `Process`；`Process` 由 block index 选择一个 `(b,c)` 序列，再执行 `CopyIn → ComputeDft → CopyOut`。source96 的 kernel 入口还保留了 `KERNEL_TYPE_MIX_AIC_1_1` 声明。本页描述这条静态调用链，不将其登记为完整 Operator producer 结果。

原件缺少 `L>0`、整数乘法范围、设备类型、UB 容量和长度上界检查。采用前必须独立确认这些条件；不能从 Python 可接收的 dtype 或模板分支推出低精度 kernel 已被这条调用链验证。尤其 source96 的非 FP32 `CopyIn` 分支复用了 `L*sizeof(float)` 拷贝字节数，本页保留原代码以供审阅，不把它推广为可用低精度路径。

## 对比依赖轴，而非按名字判断归约结构

令输入有效长度为 `L`，零填充后的变换长度为 `N=2L`，有效频率数为 `K=L+1`。在目标数学定义下，每个输出是 `sum(n=0..L-1, x[n]*exp(-i*2*pi*k*n/N))/N`；后半段补零不参与乘积。内联 reference 的实际调用是 `torch.fft.rfft(x_f32, n=2*L)` 后除以 `2*L`。本次没有执行 reference，也不假设其底层计算顺序等同直接 DFT。

| 源码事实 | source80 | source96 |
|---|---|---|
| 外层循环 | `k=0..L` | `n=0..L-1` |
| Vector Lane | 输入位置 `n`，数量 `A=align8(L)` | 输出频率 `k`，数量 `Q=align8(K)` |
| 乘积与累计 | 每个 k 的输入乘积进入 `ReduceSum`，结果经 `GetValue(0)` 取出 | 每个 n 的标量输入乘频率向量，以 `Add/Sub` 更新全部输出 Lane |
| 跨迭代依赖 | 不同 k 的输出独立；每个 k 内由实际 `ReduceSum` 重载决定归约 | 同一频率 Lane 的实、虚累计跨 n 迭代相依 |
| 归一化位置 | 每次 `ReduceSum` 的标量结果乘 `1/N` | 完成全部 n 后对输出向量乘 `1/N` |
| 输入保存 | 输入队列复制到独立 `xCalcBuf_` 后释放队列 Tensor | `xBuf_` 在 n 循环期间供标量读取 |

两份代码都直接计算 DFT，乘积工作量随 `L*K` 增长；上表只描述源码组织，不给出吞吐排序。`ReduceSum` 的具体内部累计顺序依赖目标 SDK 与重载，本页没有恢复该实现，不能把 API 名直接改写为“完整树形归约”保证。改变向量化轴、累计顺序、归一化时点和三角函数输入都会改变舍入路径，无法仅凭结构决定误差优劣。

source80 是 k 外层/n Lane，source96 是 n 外层/k Lane，并改用跨 n 累计。两者均可能被口头概括为“按频率计算”，但该概括不能证明计算组织保持不变；应分别写出外层循环和向量轴。源码结构本身不支持精度优劣排序，本页不作该主张。

## 将 InitBuffer 请求与值的存活分开核算

以下为从 `InitBuffer` 参数求和的 FP32 每 block 请求模型，使用 `sizeof(float)=4`，不是编译器布局、物理峰值或可用 UB 的实测结果。

source80 固定 `maxL=4096`，输入向量容量为 4096 个 FP32，输出容量为 4104 个 FP32。输入队列、`xCalc/reduce/index/angles/sin/cos/tmp` 共八份 4096，两个输出各 4104，`sum` 为 16；FP32 不初始化 `rawQueue_`。请求和为 `4*(8*4096+2*4104+16)=163968` 字节。`inQueue_.FreeTensor` 结束了输入值的队列占用，但代码没有重置 TPipe 或声明分配区别名；不能由 FreeTensor 推断这份请求已与别的缓冲共享。

source96 按实际 `L` 请求 `A=align8(L)` 个输入和五份 `Q=align8(L+1)` 的 `sin/cos/tmp/real/imag`，FP32 不初始化 `rawBuf_`。请求和为 `4*(A+5*Q)` 字节。输入贯穿 n 循环，实/虚累计器贯穿整个循环；`tmp` 依次承载频率索引、角度与乘积，必须保留其使用顺序；`sin` 在实部更新之后还供虚部乘积消费。去掉 `index/angles` 对象不能独立解释全部请求变化，因为 source96 还取消了独立输入副本与 reduction scratch，并改变了输出累计方式。

| L（模型输入） | source80 请求字节 | source96 请求字节 | source80 固定缓冲容纳 L | source80 拷入/归约存在补齐 Lane |
|---|---:|---:|---|---|
| 64 | 163968 | 1696 | 是 | 否 |
| 224 | 163968 | 5536 | 是 | 否 |
| 1024 | 163968 | 24736 | 是 | 否 |
| 4096 | 163968 | 98464 | 是 | 否 |
| 4097 | 163968 | 98496 | 否 | 是 |
| 8192 | 163968 | 196768 | 否 | 否 |

该表是完整 CPU 模型的实际输出摘要，所列 L 只是模型输入，不代表设备已测 shape。请求随实际长度增长仍可超过目标资源；两份实现都没有 chunk/fallback，采用时必须依据目标 SDK、真实可用 UB、额外开销和编译结果确定合法长度，不能从“动态分配”推出任意 L 可用。

## 对齐和采样范围的失效边界

source80 的 `LoadGmToUb` 将输入长度向 32 字节取整，`ComputeDft` 又以补齐后的 `A` 参与乘积和 ReduceSum，没有显式把 `n>=L` 的 Lane 置零。`L%8!=0` 时，源码可能读取当前序列之外的数据并让其参与结果；最后一行还需核对 GM 可读范围。即使 `L<=4096` 也不能消除这一问题。`L>4096` 时输入容量已越过固定分配；输出还需单独核对 `L+1<=4104`。这些是源码边界分析，没有设备失败观测。

source96 先把 `xBuf_` 清零，以 `L*sizeof(float)` 的 `DataCopyPad` 参数搬入有效输入，只在 `n<L` 读取；频率补齐 Lane 会参与向量计算，但 CopyOut 仅发布 `K` 个 FP32。这里只核对参数与使用范围，不替目标 SDK 的搬运、同步、别名或三角函数合同背书。

历史 source80 结果文件报告 `passed_cases=10, total_cases=10`；其 `full_eval` 同时含 `full_json_cases=49` 和 `passed_cases=10, total_cases=10`。因此只保留“该历史文件报告 10/10”的范围说明，不能记成 49/49，也不能绑定为本次源码的独立复现。任务通过记录不能隔离向量化轴、Sin/Cos、DMA、缓冲分配或参考实现的贡献。

需要精度排序时，应另行固定同一输入、参考、SDK、dtype、运行环境和实际二进制，比较不同累计结构并单独隔离三角函数误差，覆盖长序列、强相消、非对齐尾部和特殊值。需要性能结论时，还需测完整 wrapper 调用成本。本次均未执行。

## 相关通用知识

- [Reduce 的有效范围、存储布局与临时空间契约](../../guides/reductions/interface_and_storage_contracts.md)：采用具体 ReduceSum 重载前核对其范围与目标产品合同。
- [多输出归约的存活缓冲预算](../../guides/reductions/live_buffer_budget.md)：将缓冲声明、存活状态与目标资源约束分别核算。

## 完整原代码单位

以下代码块从原件按完整文件或完整类边界提取，没有省略号、改名、格式化或尾空格清理。共享 wrapper、tiling 与 reference 已确认两份原件字节一致；两种 kernel 及各自 pybind/launcher 分开保留。外部 SDK 与库依赖未内联，所以这些代码块不是可独立构建的工程包。

### 共享 wrapper 完整文件

```python
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_build_dir = os.path.join(_current_dir, "kernel", "build")
if _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)

import torch
import torch.nn as nn
import _hyena_rfft_ext

class ModelNew(nn.Module):
    """
    AscendC-optimized Hyena FFT size padding + RFFT + normalize.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, channels, seqlen].
        Returns:
            tuple: (x_freq_real, x_freq_imag) each with shape [batch_size, channels, seqlen+1].
        """
        x_f32 = x.float()
        real, imag = _hyena_rfft_ext.run_hyena_rfft(x_f32)
        return real, imag
```

### 共享 Tiling 完整文件

```cpp
#ifndef HYENA_RFFT_TILING_H
#define HYENA_RFFT_TILING_H

#include <cstdint>

struct HyenaRfftTiling {
    int32_t B;
    int32_t C;
    int32_t L;
    int32_t N;
    int32_t totalSeq;
};

#endif
```

### 共享 reference 的导入与完整 Model 类

```python
import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused FFT size padding and real FFT computation for Hyena convolution.
    Pads input to 2*seqlen for circular convolution, computes real FFT (rfft),
    and normalizes by fft_size.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Applies FFT size padding and real FFT computation for Hyena convolution.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, channels, seqlen].
                              Supports float32.

        Returns:
            tuple: (x_freq_real, x_freq_imag)
                - x_freq_real (torch.Tensor): Real part of normalized frequency domain output
                                              with shape [batch_size, channels, seqlen+1].
                - x_freq_imag (torch.Tensor): Imaginary part of normalized frequency domain output
                                              with shape [batch_size, channels, seqlen+1].
        """
        batch, channels, seqlen = x.shape
        fft_size = 2 * seqlen

        x_f32 = x.to(torch.float32)

        x_freq = torch.fft.rfft(x_f32, n=fft_size)

        x_freq = x_freq / fft_size

        x_freq_real = x_freq.real.contiguous()
        x_freq_imag = x_freq.imag.contiguous()

        return x_freq_real, x_freq_imag


```

### source80 pybind 完整文件

```cpp
#include <algorithm>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "hyena_rfft_tiling.h"

extern "C" void hyena_rfft_do_fp32(
    uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* real, uint8_t* imag, uint8_t* tiling);

extern "C" void hyena_rfft_do_fp16(
    uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* real, uint8_t* imag, uint8_t* tiling);

extern "C" void hyena_rfft_do_bf16(
    uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* real, uint8_t* imag, uint8_t* tiling);

namespace hyena_rfft_ext {

using LaunchFn = void (*)(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*);

pybind11::tuple run_hyena_rfft(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() == 3, "x must be [B, C, L]");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
        "x must be float16, float32, or bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int32_t B = static_cast<int32_t>(x.sizes()[0]);
    int32_t C = static_cast<int32_t>(x.sizes()[1]);
    int32_t L = static_cast<int32_t>(x.sizes()[2]);
    int32_t N = 2 * L;
    int32_t totalSeq = B * C;

    auto options = x.options().dtype(at::kFloat);
    at::Tensor real = at::empty({B, C, L + 1}, options);
    at::Tensor imag = at::empty({B, C, L + 1}, options);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(HyenaRfftTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto* tiling = reinterpret_cast<HyenaRfftTiling*>(tilingCpu.data_ptr());
    tiling->B = B;
    tiling->C = C;
    tiling->L = L;
    tiling->N = N;
    tiling->totalSeq = totalSeq;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (x.scalar_type() == at::kFloat) {
        launch = hyena_rfft_do_fp32;
    } else if (x.scalar_type() == at::kHalf) {
        launch = hyena_rfft_do_fp16;
    } else if (x.scalar_type() == at::kBFloat16) {
        launch = hyena_rfft_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        totalSeq,
        aclStream,
        static_cast<uint8_t*>(const_cast<void*>(x.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(real.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(imag.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(tilingNpu.storage().data())));

    return pybind11::make_tuple(real, imag);
}

}  // namespace hyena_rfft_ext

PYBIND11_MODULE(_hyena_rfft_ext, m)
{
    m.doc() = "hyena_rfft extension";
    m.def("run_hyena_rfft", &hyena_rfft_ext::run_hyena_rfft, "");
}
```

### source80 launcher 完整文件

```cpp
#include "hyena_rfft_kernel.h"

extern "C" __global__ __aicore__ void hyena_rfft_custom_fp32(
    GM_ADDR x, GM_ADDR real, GM_ADDR imag, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    HyenaRfftKernel<float> kernel;
    kernel.Init(x, real, imag, tiling);
    kernel.Process();
}

extern "C" __global__ __aicore__ void hyena_rfft_custom_fp16(
    GM_ADDR x, GM_ADDR real, GM_ADDR imag, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    HyenaRfftKernel<half> kernel;
    kernel.Init(x, real, imag, tiling);
    kernel.Process();
}

extern "C" __global__ __aicore__ void hyena_rfft_custom_bf16(
    GM_ADDR x, GM_ADDR real, GM_ADDR imag, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    HyenaRfftKernel<bfloat16_t> kernel;
    kernel.Init(x, real, imag, tiling);
    kernel.Process();
}

extern "C" void hyena_rfft_do_fp32(
    uint32_t blockDim,
    void* stream,
    uint8_t* x,
    uint8_t* real,
    uint8_t* imag,
    uint8_t* tiling)
{
    hyena_rfft_custom_fp32<<<blockDim, nullptr, stream>>>(x, real, imag, tiling);
}

extern "C" void hyena_rfft_do_fp16(
    uint32_t blockDim,
    void* stream,
    uint8_t* x,
    uint8_t* real,
    uint8_t* imag,
    uint8_t* tiling)
{
    hyena_rfft_custom_fp16<<<blockDim, nullptr, stream>>>(x, real, imag, tiling);
}

extern "C" void hyena_rfft_do_bf16(
    uint32_t blockDim,
    void* stream,
    uint8_t* x,
    uint8_t* real,
    uint8_t* imag,
    uint8_t* tiling)
{
    hyena_rfft_custom_bf16<<<blockDim, nullptr, stream>>>(x, real, imag, tiling);
}
```

### source80 kernel 完整文件

```cpp
#pragma once

#include <kernel_operator.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>

#include "hyena_rfft_tiling.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

template <typename T>
__aicore__ inline void CopyTilingT(HyenaRfftTiling* tiling, GM_ADDR tilingGM) {
    int32_t* dst = reinterpret_cast<int32_t*>(tiling);
    auto* src = reinterpret_cast<__gm__ int32_t*>(tilingGM);
    for (size_t i = 0; i < sizeof(HyenaRfftTiling) / sizeof(int32_t); ++i) {
        dst[i] = src[i];
    }
}

template <typename T>
__aicore__ inline uint32_t AlignUpBytes(uint32_t bytes) {
    return ((bytes + 31) / 32) * 32;
}

template <typename T>
__aicore__ inline void LoadGmToUb(AscendC::LocalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, uint32_t count) {
    uint32_t elemSize = static_cast<uint32_t>(sizeof(T));
    uint32_t alignedCount = AlignUpBytes<T>(count * elemSize) / elemSize;
    AscendC::DataCopy(dst, src, alignedCount);
}

template <typename T>
__aicore__ inline void StoreUbToGm(const AscendC::GlobalTensor<T>& dst, const AscendC::LocalTensor<T>& src, uint32_t count) {
    AscendC::DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
}

template <typename T>
class HyenaRfftKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR real, GM_ADDR imag, GM_ADDR tilingGM) {
        CopyTilingT<T>(&tiling_, tilingGM);
        int64_t xSize = (int64_t)tiling_.B * tiling_.C * tiling_.L;
        int64_t outSize = (int64_t)tiling_.B * tiling_.C * (tiling_.L + 1);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), xSize);
        realGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(real), outSize);
        imagGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(imag), outSize);

        int32_t maxL = 4096;
        int32_t maxInLen = AlignUpBytes<float>(maxL * sizeof(float)) / sizeof(float);
        int32_t maxOutLen = ((maxL + 1 + 7) / 8) * 8;

        pipe_.InitBuffer(inQueue_, 1, maxInLen * sizeof(float));
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(rawQueue_, 1, maxInLen * sizeof(T));
        }
        pipe_.InitBuffer(xCalcBuf_, maxInLen * sizeof(float));
        pipe_.InitBuffer(reduceBuf_, maxInLen * sizeof(float));
        pipe_.InitBuffer(sumBuf_, 16 * sizeof(float));
        pipe_.InitBuffer(outQueueReal_, 1, maxOutLen * sizeof(float));
        pipe_.InitBuffer(outQueueImag_, 1, maxOutLen * sizeof(float));
        pipe_.InitBuffer(indexBuf_, maxInLen * sizeof(float));
        pipe_.InitBuffer(anglesBuf_, maxInLen * sizeof(float));
        pipe_.InitBuffer(sinBuf_, maxInLen * sizeof(float));
        pipe_.InitBuffer(cosBuf_, maxInLen * sizeof(float));
        pipe_.InitBuffer(tmpBuf_, maxInLen * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t m = AscendC::GetBlockIdx();
        if (m >= tiling_.totalSeq) {
            return;
        }
        int32_t b = m / tiling_.C;
        int32_t c = m % tiling_.C;

        CopyIn(b, c);
        ComputeDft();
        CopyOut(b, c);
    }

private:
    __aicore__ inline void CopyIn(int32_t b, int32_t c) {
        int32_t L = tiling_.L;
        int64_t offset = ((int64_t)b * tiling_.C + c) * L;

        if constexpr (std::is_same_v<T, float>) {
            inQueue_.AllocTensor<float>(xLocal_);
            LoadGmToUb<float>(xLocal_, xGM_[offset], L);
            inQueue_.EnQue(xLocal_);
        } else {
            rawQueue_.AllocTensor<T>(rawLocal_);
            LoadGmToUb<T>(rawLocal_, xGM_[offset], L);
            rawQueue_.EnQue(rawLocal_);
            rawQueue_.DeQue<T>(rawLocal_);
            inQueue_.AllocTensor<float>(xLocal_);
            AscendC::Cast(xLocal_, rawLocal_, AscendC::RoundMode::CAST_NONE, L);
            inQueue_.EnQue(xLocal_);
            rawQueue_.FreeTensor(rawLocal_);
        }
    }

    __aicore__ inline void ComputeDft() {
        int32_t L = tiling_.L;
        int32_t N = tiling_.N;
        float norm = 1.0f / (float)N;
        float two_pi_over_N = 2.0f * M_PI_F / (float)N;
        uint32_t alignedL = AlignUpBytes<float>(L * sizeof(float)) / sizeof(float);

        inQueue_.DeQue<float>(xLocal_);

        auto xCalcLocal = xCalcBuf_.Get<float>();
        AscendC::DataCopy(xCalcLocal, xLocal_, alignedL);
        AscendC::PipeBarrier<PIPE_ALL>();
        inQueue_.FreeTensor(xLocal_);

        outQueueReal_.AllocTensor<float>(realOutLocal_);
        outQueueImag_.AllocTensor<float>(imagOutLocal_);

        auto indexLocal = indexBuf_.Get<float>();
        auto anglesLocal = anglesBuf_.Get<float>();
        auto sinLocal = sinBuf_.Get<float>();
        auto cosLocal = cosBuf_.Get<float>();
        auto tmpLocal = tmpBuf_.Get<float>();
        auto reduceLocal = reduceBuf_.Get<float>();
        auto sumLocal = sumBuf_.Get<float>();

        for (int32_t k = 0; k <= L; ++k) {
            float baseAngle = two_pi_over_N * (float)k;

            AscendC::CreateVecIndex(indexLocal, 0.0f, alignedL);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(anglesLocal, indexLocal, baseAngle, alignedL);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cos(cosLocal, anglesLocal, alignedL);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(tmpLocal, xCalcLocal, cosLocal, alignedL);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float>(sumLocal, tmpLocal, reduceLocal, alignedL);
            AscendC::PipeBarrier<PIPE_V>();
            float realVal = sumLocal.GetValue(0) * norm;

            AscendC::Sin(sinLocal, anglesLocal, alignedL);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(tmpLocal, xCalcLocal, sinLocal, alignedL);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float>(sumLocal, tmpLocal, reduceLocal, alignedL);
            AscendC::PipeBarrier<PIPE_V>();
            float imagVal = -sumLocal.GetValue(0) * norm;

            realOutLocal_.SetValue(k, realVal);
            imagOutLocal_.SetValue(k, imagVal);
        }

        outQueueReal_.EnQue(realOutLocal_);
        outQueueImag_.EnQue(imagOutLocal_);
    }

    __aicore__ inline void CopyOut(int32_t b, int32_t c) {
        int32_t L = tiling_.L;
        int64_t offset = ((int64_t)b * tiling_.C + c) * (L + 1);

        outQueueReal_.DeQue<float>(realOutLocal_);
        outQueueImag_.DeQue<float>(imagOutLocal_);

        StoreUbToGm<float>(realGM_[offset], realOutLocal_, L + 1);
        AscendC::PipeBarrier<PIPE_MTE2>();
        StoreUbToGm<float>(imagGM_[offset], imagOutLocal_, L + 1);
        AscendC::PipeBarrier<PIPE_MTE2>();

        outQueueReal_.FreeTensor(realOutLocal_);
        outQueueImag_.FreeTensor(imagOutLocal_);
    }

    HyenaRfftTiling tiling_;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> rawQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCalcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueueReal_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueueImag_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> anglesBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sinBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cosBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<T> rawLocal_;
    AscendC::LocalTensor<float> realOutLocal_;
    AscendC::LocalTensor<float> imagOutLocal_;
    AscendC::GlobalTensor<T> xGM_;
    AscendC::GlobalTensor<float> realGM_;
    AscendC::GlobalTensor<float> imagGM_;
};
```

### source96 pybind 完整文件

```cpp
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "hyena_rfft_tiling.h"

extern "C" void hyena_rfft_do_fp32(
    uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* real, uint8_t* imag, uint8_t* tiling);

namespace hyena_rfft_ext {

using LaunchFn = void (*)(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*);

pybind11::tuple run_hyena_rfft(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() == 3, "x must be [B, C, L]");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int32_t B = static_cast<int32_t>(x.sizes()[0]);
    int32_t C = static_cast<int32_t>(x.sizes()[1]);
    int32_t L = static_cast<int32_t>(x.sizes()[2]);
    int32_t N = 2 * L;
    int32_t totalSeq = B * C;

    auto options = x.options().dtype(at::kFloat);
    at::Tensor real = at::empty({B, C, L + 1}, options);
    at::Tensor imag = at::empty({B, C, L + 1}, options);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(HyenaRfftTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto* tiling = reinterpret_cast<HyenaRfftTiling*>(tilingCpu.data_ptr());
    tiling->B = B;
    tiling->C = C;
    tiling->L = L;
    tiling->N = N;
    tiling->totalSeq = totalSeq;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = hyena_rfft_do_fp32;

    launch(
        totalSeq,
        aclStream,
        static_cast<uint8_t*>(const_cast<void*>(x.data_ptr())),
        static_cast<uint8_t*>(const_cast<void*>(real.data_ptr())),
        static_cast<uint8_t*>(const_cast<void*>(imag.data_ptr())),
        static_cast<uint8_t*>(const_cast<void*>(tilingNpu.data_ptr())));

    return pybind11::make_tuple(real, imag);
}

}  // namespace hyena_rfft_ext

PYBIND11_MODULE(_hyena_rfft_ext, m)
{
    m.doc() = "hyena_rfft extension";
    m.def("run_hyena_rfft", &hyena_rfft_ext::run_hyena_rfft, "");
}
```

### source96 launcher 完整文件

```cpp
#include "hyena_rfft_kernel.h"

extern "C" __global__ __aicore__ void hyena_rfft_custom_fp32(
    GM_ADDR x, GM_ADDR real, GM_ADDR imag, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::TPipe pipe;
    HyenaRfftKernel<float> kernel;
    kernel.Init(x, real, imag, tiling);
    kernel.Process();
}

extern "C" void hyena_rfft_do_fp32(
    uint32_t blockDim,
    void* stream,
    uint8_t* x,
    uint8_t* real,
    uint8_t* imag,
    uint8_t* tiling)
{
    hyena_rfft_custom_fp32<<<blockDim, nullptr, stream>>>(x, real, imag, tiling);
}
```

### source96 kernel 完整文件

```cpp
#pragma once
#include <kernel_operator.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>
#include "hyena_rfft_tiling.h"
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif
__aicore__ inline void CopyTiling(HyenaRfftTiling* tiling, GM_ADDR tilingGM) {
    int32_t* dst = reinterpret_cast<int32_t*>(tiling);
    auto* src = reinterpret_cast<__gm__ int32_t*>(tilingGM);
    for (size_t i = 0; i < sizeof(HyenaRfftTiling) / sizeof(int32_t); ++i) dst[i] = src[i];
}
__aicore__ inline int32_t AlignUp(int32_t n, int32_t align) { return (n + align - 1) / align * align; }
template <typename T>
class HyenaRfftKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR real, GM_ADDR imag, GM_ADDR tilingGM) {
        CopyTiling(&tiling_, tilingGM);
        int64_t xSize = (int64_t)tiling_.B * tiling_.C * tiling_.L;
        int64_t outSize = (int64_t)tiling_.B * tiling_.C * (tiling_.L + 1);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), xSize);
        realGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(real), outSize);
        imagGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(imag), outSize);
        int32_t bufL = AlignUp(tiling_.L, 8);
        int32_t outBufLen = AlignUp(tiling_.L + 1, 8);
        pipe_.InitBuffer(xBuf_, bufL * sizeof(float));
        if constexpr (!std::is_same_v<T, float>) pipe_.InitBuffer(rawBuf_, bufL * sizeof(T));
        pipe_.InitBuffer(sinBuf_, outBufLen * sizeof(float));
        pipe_.InitBuffer(cosBuf_, outBufLen * sizeof(float));
        pipe_.InitBuffer(tmpBuf_, outBufLen * sizeof(float));
        pipe_.InitBuffer(realOutBuf_, outBufLen * sizeof(float));
        pipe_.InitBuffer(imagOutBuf_, outBufLen * sizeof(float));
    }
    __aicore__ inline void Process() {
        int32_t m = AscendC::GetBlockIdx();
        if (m >= tiling_.totalSeq) return;
        int32_t b = m / tiling_.C;
        int32_t c = m % tiling_.C;
        CopyIn(b, c);
        ComputeDft();
        CopyOut(b, c);
    }
private:
    __aicore__ inline void CopyIn(int32_t b, int32_t c) {
        int32_t L = tiling_.L;
        int32_t bufL = AlignUp(L, 8);
        int64_t offset = ((int64_t)b * tiling_.C + c) * L;
        auto xLocal = xBuf_.Get<float>();
        AscendC::Duplicate(xLocal, 0.0f, bufL);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(L * sizeof(float)), 0, 0, 0};
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopyPad(xLocal, xGM_[offset], copyParams, AscendC::DataCopyPadExtParams<float>{true, 0, 0, 0.0f});
            AscendC::PipeBarrier<PIPE_MTE2>();
        } else {
            auto rawLocal = rawBuf_.Get<T>();
            AscendC::DataCopyPad(rawLocal, xGM_[offset], copyParams, AscendC::DataCopyPadExtParams<T>{true, 0, 0, (T)0});
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::Cast(xLocal, rawLocal, AscendC::RoundMode::CAST_NONE, L);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void ComputeDft() {
        int32_t L = tiling_.L;
        int32_t N = tiling_.N;
        int32_t outLen = L + 1;
        int32_t outBufLen = AlignUp(outLen, 8);
        float norm = 1.0f / (float)N;
        float two_pi_over_N = 2.0f * M_PI_F / (float)N;
        auto xLocal = xBuf_.Get<float>();
        auto sinLocal = sinBuf_.Get<float>();
        auto cosLocal = cosBuf_.Get<float>();
        auto tmpLocal = tmpBuf_.Get<float>();
        auto realOutLocal = realOutBuf_.Get<float>();
        auto imagOutLocal = imagOutBuf_.Get<float>();
        AscendC::Duplicate(realOutLocal, 0.0f, outBufLen);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(imagOutLocal, 0.0f, outBufLen);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t n = 0; n < L; ++n) {
            float xn = xLocal.GetValue(n);
            float baseAngle = two_pi_over_N * (float)n;
            AscendC::CreateVecIndex(tmpLocal, 0.0f, outBufLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(tmpLocal, tmpLocal, baseAngle, outBufLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sin(sinLocal, tmpLocal, outBufLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cos(cosLocal, tmpLocal, outBufLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(tmpLocal, cosLocal, xn, outBufLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(realOutLocal, realOutLocal, tmpLocal, outBufLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(tmpLocal, sinLocal, xn, outBufLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(imagOutLocal, imagOutLocal, tmpLocal, outBufLen);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Muls(realOutLocal, realOutLocal, norm, outBufLen);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(imagOutLocal, imagOutLocal, norm, outBufLen);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CopyOut(int32_t b, int32_t c) {
        int32_t L = tiling_.L;
        int32_t outLen = L + 1;
        int64_t offset = ((int64_t)b * tiling_.C + c) * outLen;
        auto realOutLocal = realOutBuf_.Get<float>();
        auto imagOutLocal = imagOutBuf_.Get<float>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(outLen * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(realGM_[offset], realOutLocal, copyParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::DataCopyPad(imagGM_[offset], imagOutLocal, copyParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    HyenaRfftTiling tiling_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECIN> xBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sinBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cosBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> realOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> imagOutBuf_;
    AscendC::GlobalTensor<T> xGM_;
    AscendC::GlobalTensor<float> realGM_;
    AscendC::GlobalTensor<float> imagGM_;
};
```
