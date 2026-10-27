---
type: "Runbook"
title: Float-AR ArgMax：ReduceMax calIndex 分块实现
description: 最内轴 fp32 argmax 的完整 AscendC kernel 配方：DataCopyPad 分块搬入、ReduceMax(calIndex=true) 逐块归约（索引按 dst float 位模式存储须 reinterpret 读取）、标量严格大于合并保首最大值、Cast float→int64 单元素输出。验证等级为编译级 + 宿主 UT，未上板。
tags: [task.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:18:45Z'
updated_at: '2026-09-16T10:18:45Z'
platforms: ["950"]
---

# Float-AR ArgMax：ReduceMax calIndex 分块实现

## 现象

在 Ascend950PR（arch35/DAV_3510，CANN 9.1.0）上从零实现 argmax 算子（bench_no_harness_arg_max，输出 int64 索引，多最大值返回第一个）时，最常见且性能关键的形态是 dim 为最内轴（AR 模式：每行 R 个连续元素归约成 1 个索引）。该路径需要一套完整的"分块搬入 → UB 内归约取最大值+索引 → 跨块标量合并 → float 索引转 int64 → 单元素搬运出"的 kernel 配方，任何一环（索引读取方式、合并语义、Cast 时机、搬运字节数）出错都会破坏 bitwise-equal。本卡给出该配方在首轮迭代中通过完整编译验证的实现（5 个 ascend950 kernel 二进制生成成功），并记录选型时踩到的 API 约束。

## 适用场景

- 工程形态：registry-invoke 自定义算子（手写 AscendC kernel，TilingKey 模板分发，仅 arch35 目录）。
- API 路径：`ReduceMax` Level 2 接口（`ReduceMax(dst, src, sharedTmpBuffer, count, calIndex)`）、`DataCopyPad`/`DataCopyExtParams`/`DataCopyPadExtParams`、`Cast<int64_t, float>`、`TQue<TPosition::VECIN>`/`TBuf<TPosition::VECCALC>`、`GET_TILING_DATA_WITH_STRUCT`。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（--soc=ascend950 完整编译通过；本卡范围内未上板运行）。
- 适用粒度：operator_family（归约含索引跟踪类算子：argmax/argmin/max-with-index 的最内轴 float 路径）。
- 路线优先级：recommended_fastpath（AR+float 是 argmax 首选路径；ARA/int 路线的约束另见失效边界）。
- 明确不适用/未验证场景：归约轴非最内轴（ARA 模式）；int32/int64/bf16 输入（首轮为 stub）；上板精度与性能未验证。

## 根因/解决配方

配方分四段（完整代码见下）：

1. **缓冲布局**：单 TQue（VECIN，rAlign * sizeof(float)）+ 三个 VECCALC TBuf（dstBuf 32B 存 max+idx、tmpBuf 为 ReduceMax sharedTmpBuffer、idxBuf/outBuf 各 32B）。
2. **逐 chunk 归约**：`ReduceMax<float>(dstLocal, inData, tmpLocal, chunkLen, true)`；`dstLocal.GetValue(0)` 是最大值，`GetValue(1)` 是索引——索引按 dst 的 float 位模式存储，必须 `reinterpret_cast<const uint32_t*>` 读出（官方文档：按 half/float 格式直接读索引值是错的）。
3. **跨 chunk 标量合并（首最大值语义）**：`if (chunkMax > globalMax)` 严格大于才更新——与 ReduceMax(calIndex=true) 文档"多最大值返回第一个"对齐；全局索引 = `chunkStart + chunkArgIdx`。
4. **输出**：`idxLocal.SetValue(0, (float)globalIdx)` → `PipeBarrier<PIPE_ALL>()` → `Cast<int64_t, float>(outLocal, idxLocal, RoundMode::CAST_ROUND, 1)` → `PipeBarrier<PIPE_ALL>()` → `DataCopyPad` 单元素 8 字节出 GM（blockLen = sizeof(int64_t)）。

关键约束（选型时由官方 API 文档核实，A3 产品系列行）：
- `ReduceMax` 模板 T 仅支持 half/float（A3）；bf16/int32/int64 不能直接用。
- half 输入时索引上限 65535（uint16）；R 可能到 2^21，因此 fp16 输入也应先 Cast→float 再 ReduceMax（本迭代只实现 fp32 直用，fp16/bf16 为后续计划）。
- `DataCopyPad` 支持 int64（A3），blockCount∈[1,4095]、blockLen∈[1,2097151] 字节；GM 侧 stride 单位是字节。

## 设计决策

- **为什么用 ReduceMax(calIndex) 而不是自建比较循环**：Level 2 接口一次给出值+索引且文档保证首最大值，官方支持 half/float 正好覆盖本路径；自建 Compare+Select 循环留给 ARA 模式（那里 Pattern 接口不返回索引）。
- **为什么跨 chunk 用标量 `>` 而不是 `>=`**：`>=` 会把"后块等值"也更新，破坏首最大值语义；ReduceMax 块内已保证首最大值，块间必须同样严格。
- **为什么 chunk 内直接复用 inQueue 张量**：ReduceMax 的 src 直接吃 VECIN DeQue 出的 LocalTensor，无需额外拷贝；dst/tmp 用 VECCALC TBuf 常驻。
- **tmpBufSize 取 max(8KB, rAlign*4B) 再 32B 对齐**：官方要求 calIndex=true 时 sharedTmpBuffer 累加各轮空间且 32 字节对齐；8KB 下限是保守值（保守取整轮 UB 能装下的 chunk 长度）。
- **放弃的候选**：ReduceMax-35（Pattern 版）不支持 calIndex，无法拿索引，弃用；把 chunk 结果写回 GM 再二次归约——多一次 GM 往返，弃用。
- **路线归属**：recommended_fastpath（AR+float 为 argmax 主路径）。

## 系统信息

- 工具链：CANN 9.1.0（`$ASCEND_HOME_PATH` 指向安装根），cmake + opbuild，`bash build.sh --soc=ascend950 -j8`。
- 目标平台：Ascend950PR（arch35/DAV_3510），仅 arch35 工程目录。
- 目录布局：`operators/bench_no_harness_arg_max/{op_host,op_kernel,op_api,op_graph,tests}`。
- 相关文件（与事件流实际读写一一对应）：
  - `op_kernel/arch35/bench_no_harness_arg_max.h`（kernel 类，本卡完整代码）
  - `op_kernel/bench_no_harness_arg_max_arch35.cpp`（kernel 入口，本卡完整代码）
  - `op_kernel/arch35/bench_no_harness_arg_max_tiling_data.h`、`..._tiling_key.h`
  - `op_host/arch35/bench_no_harness_arg_max_tiling.cpp`（AR 模式 tiling：rAlign/ubFactorR/tmpBufSize 计算）

## 完整代码

kernel 类（`op_kernel/arch35/bench_no_harness_arg_max.h`，首轮终版，verbatim；`-std::numeric_limits...` 行已按编译修复改为 `NumericLimits<float>::Infinity()`）：

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the CANN Open Software License Agreement Version 2.0.
 */

/*!
 * \file bench_no_harness_arg_max.h
 * \brief BenchNoHarnessArgMax Kernel class (arch35)
 *
 * Design ref: DESIGN.md §3.7.1 (Float-AR), §3.10 (UB budget)
 *
 * Iter1: Float-AR path (fp32, dim=innermost, A0=1)
 *   Data flow: GM → DataCopyPad → UB(float) → ReduceMax(calIndex=true)
 *              → scalar merge → Cast(float→int64) → DataCopyPad → GM
 *
 * Other paths: stub (iter2/3 will implement)
 */

#ifndef BENCH_NO_HARNESS_ARG_MAX_H
#define BENCH_NO_HARNESS_ARG_MAX_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bench_no_harness_arg_max_tiling_data.h"
#include "bench_no_harness_arg_max_tiling_key.h"

namespace NsBenchNoHarnessArgMax {

using namespace AscendC;

template <typename D_T_X, uint32_t MODE>
class BenchNoHarnessArgMax {
    static constexpr int32_t BUFFER_NUM = 1;  // iter1: single buffer

public:
    __aicore__ inline BenchNoHarnessArgMax() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR indices,
                                const BenchNoHarnessArgMaxTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessFloatAR();

private:
    TPipe pipe;

    // Input queue: float data (after Cast if needed; fp32 direct)
    TQue<QuePosition::VECIN, 1> inQueueX;

    // Temporary buffers
    TBuf<TPosition::VECCALC> dstBuf;    // ReduceMax output (max + index)
    TBuf<TPosition::VECCALC> tmpBuf;    // ReduceMax sharedTmpBuffer
    TBuf<TPosition::VECCALC> idxBuf;    // float index for Cast
    TBuf<TPosition::VECCALC> outBuf;    // int64 output

    GlobalTensor<D_T_X> inputGM;
    GlobalTensor<int64_t> indicesGM;

    // Tiling parameters
    uint64_t aSize_      = 0;  // A1
    uint64_t rSize_      = 0;  // R
    uint64_t nextASize_  = 0;  // A0
    uint32_t blkFactor_  = 0;
    uint32_t ubFactorR_  = 0;
    uint32_t rAlign_     = 0;
    uint32_t tmpBufSize_ = 0;
    int64_t  rowStart_   = 0;
    int64_t  rowEnd_     = 0;
};

template <typename D_T_X, uint32_t MODE>
__aicore__ inline void BenchNoHarnessArgMax<D_T_X, MODE>::Init(
    GM_ADDR input, GM_ADDR indices, const BenchNoHarnessArgMaxTilingData* tilingData)
{
    aSize_     = tilingData->aSize;
    rSize_     = tilingData->rSize;
    nextASize_ = tilingData->nextASize;
    blkFactor_ = tilingData->blkFactor;
    ubFactorR_ = tilingData->ubFactorR;
    rAlign_    = tilingData->rAlign;
    tmpBufSize_ = tilingData->tmpBufSize;

    // Compute row range for this core
    uint32_t blockIdx = GetBlockIdx();
    uint64_t totalOutput = aSize_ * nextASize_;  // A1 * A0
    rowStart_ = static_cast<int64_t>(blockIdx) * blkFactor_;
    int64_t rowsThisCore = static_cast<int64_t>(blkFactor_);
    if (static_cast<uint64_t>(rowStart_) + rowsThisCore > totalOutput) {
        rowsThisCore = static_cast<int64_t>(totalOutput) - rowStart_;
    }
    if (rowsThisCore < 0) {
        rowsThisCore = 0;
    }
    rowEnd_ = rowStart_ + rowsThisCore;

    // Set up GlobalTensor
    inputGM.SetGlobalBuffer((__gm__ D_T_X*)input, tilingData->inputTotal);
    indicesGM.SetGlobalBuffer((__gm__ int64_t*)indices, totalOutput);

    // Initialize UB buffers (only needed for Float-AR path)
    if constexpr (std::is_same_v<D_T_X, float> && MODE == ARG_MAX_MODE_AR) {
        // Input queue: rAlign * sizeof(float) per buffer
        pipe.InitBuffer(inQueueX, BUFFER_NUM, rAlign_ * sizeof(float));
        // ReduceMax output: 2 floats, padded to 32B
        pipe.InitBuffer(dstBuf, 32);
        // sharedTmpBuffer for ReduceMax
        pipe.InitBuffer(tmpBuf, tmpBufSize_);
        // float index buffer
        pipe.InitBuffer(idxBuf, 32);
        // int64 output buffer
        pipe.InitBuffer(outBuf, 32);
    }
}

template <typename D_T_X, uint32_t MODE>
__aicore__ inline void BenchNoHarnessArgMax<D_T_X, MODE>::ProcessFloatAR()
{
    // Buffers
    LocalTensor<float> dstLocal = dstBuf.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    LocalTensor<float> idxLocal = idxBuf.Get<float>();

    for (int64_t row = rowStart_; row < rowEnd_; row++) {
        float globalMax = -NumericLimits<float>::Infinity();
        uint32_t globalIdx = 0;

        uint64_t rowOffset = static_cast<uint64_t>(row) * rSize_;

        // Chunk loop
        uint32_t numChunks = (rSize_ + ubFactorR_ - 1) / ubFactorR_;
        for (uint32_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
            uint64_t chunkStart = static_cast<uint64_t>(chunkIdx) * ubFactorR_;
            uint32_t chunkLen = static_cast<uint32_t>(rSize_ - chunkStart);
            if (chunkLen > ubFactorR_) {
                chunkLen = ubFactorR_;
            }

            // CopyIn: GM → UB
            LocalTensor<float> inLocal = inQueueX.template AllocTensor<float>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = chunkLen * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            DataCopyPad(inLocal, inputGM[rowOffset + chunkStart], copyParams, padParams);
            inQueueX.EnQue(inLocal);

            // Compute: ReduceMax with calIndex
            LocalTensor<float> inData = inQueueX.template DeQue<float>();
            ReduceMax<float>(dstLocal, inData, tmpLocal,
                             static_cast<int32_t>(chunkLen), true);

            // Extract results
            float chunkMax = dstLocal.GetValue(0);
            float idxFloat = dstLocal.GetValue(1);
            uint32_t chunkArgIdx;
            // Reinterpret float bits as uint32_t (index stored as float bit pattern)
            const float* idxPtr = &idxFloat;
            chunkArgIdx = *reinterpret_cast<const uint32_t*>(idxPtr);

            // Scalar merge: first-max semantics (only update if strictly greater)
            if (chunkMax > globalMax) {
                globalMax = chunkMax;
                globalIdx = static_cast<uint32_t>(chunkStart) + chunkArgIdx;
            }

            inQueueX.FreeTensor(inData);
        }

        // CopyOut: Cast float index → int64, then DataCopyPad to GM
        idxLocal.SetValue(0, static_cast<float>(globalIdx));
        PipeBarrier<PIPE_ALL>();

        LocalTensor<int64_t> outLocal = outBuf.Get<int64_t>();
        Cast<int64_t, float>(outLocal, idxLocal, RoundMode::CAST_ROUND, 1);
        PipeBarrier<PIPE_ALL>();

        DataCopyExtParams outCopyParams;
        outCopyParams.blockCount = 1;
        outCopyParams.blockLen = sizeof(int64_t);  // 8 bytes
        outCopyParams.srcStride = 0;
        outCopyParams.dstStride = 0;
        DataCopyPad(indicesGM[static_cast<uint64_t>(row)], outLocal, outCopyParams);
    }
}

template <typename D_T_X, uint32_t MODE>
__aicore__ inline void BenchNoHarnessArgMax<D_T_X, MODE>::Process()
{
    if constexpr (std::is_same_v<D_T_X, float> && MODE == ARG_MAX_MODE_AR) {
        // Float-AR path: fully implemented (iter1)
        ProcessFloatAR();
    } else {
        // Other paths: not implemented in iter1
        // iter2: Float-ARA, Float16-AR/ARA, BFloat16-AR/ARA
        // iter3: Int32-AR/ARA, Int64-AR/ARA
    }
}

} // namespace NsBenchNoHarnessArgMax
#endif // BENCH_NO_HARNESS_ARG_MAX_H
```

kernel 入口（`op_kernel/bench_no_harness_arg_max_arch35.cpp`，verbatim）：

```cpp
#include "arch35/bench_no_harness_arg_max.h"

template <typename D_T_X, uint32_t MODE>
__global__ __aicore__ void bench_no_harness_arg_max(
    GM_ADDR input, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessArgMaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessArgMaxTilingData, tilingData, tiling);
    NsBenchNoHarnessArgMax::BenchNoHarnessArgMax<D_T_X, MODE> op;
    op.Init(input, indices, &tilingData);
    op.Process();
}
```

AR 模式 tiling 要点（`op_host/arch35/bench_no_harness_arg_max_tiling.cpp`，关键段落 verbatim）：

```cpp
    if (mode == ARG_MAX_MODE_AR) {
        // rAlign: R aligned to 32B in elements (for Buffer allocation)
        // For float: 32B / 4B = 8 elements
        uint32_t alignElements = 32 / dtypeSize;
        tiling->rAlign = static_cast<uint32_t>(CeilAlign(r, static_cast<uint64_t>(alignElements)));

        // tmpBufSize for ReduceMax(calIndex=true) sharedTmpBuffer
        // Conservative: max(8KB, rAlign * dtypeSize)
        uint32_t tmpBufSize = MIN_TMP_BUF_SIZE;
        uint32_t rAlignBytes = tiling->rAlign * dtypeSize;
        if (rAlignBytes > tmpBufSize) {
            tmpBufSize = rAlignBytes;
        }
        tmpBufSize = static_cast<uint32_t>(CeilAlign(tmpBufSize, 32U));
        tiling->tmpBufSize = tmpBufSize;

        // ubFactorR: max R elements that fit in UB (after overhead)
        // UB budget: inBuf(rAlign * dtypeSize) + dstBuf(32) + tmpBuf(tmpBufSize) + idxBuf(32) + outBuf(32)
        uint64_t overhead = UB_OVERHEAD + tmpBufSize;
        uint64_t availableForInput = (ubSize > overhead) ? (ubSize - overhead) : 0;
        uint64_t maxUbFactorR = availableForInput / dtypeSize;
        uint32_t ubFactorR = static_cast<uint32_t>(
            FloorAlign(std::min(maxUbFactorR, static_cast<uint64_t>(r)),
                       static_cast<uint64_t>(alignElements)));
        ubFactorR = (ubFactorR == 0) ? static_cast<uint32_t>(alignElements) : ubFactorR;
        tiling->ubFactorR = ubFactorR;
        tiling->ubFactorA0 = 1;
        tiling->a0Align = 1;
    }
```

多核切分沿输出维（A1*A0 展平）均分：`blkFactor = CeilDiv(totalOutput, coreNum)`，尾核 `blkTailFactor` 收余；空张量（inputTotal==0 或 r==0）直接 BlockDim=1 早退。

## 性能影响

本卡范围内未做上板性能测量，无相对对比数据（未验证）。

## 预防

- 使用 ReduceMax(calIndex=true) 前先确认输入 dtype ∈ {half, float}；bf16/int32/int64 需先走 Cast 或替代路线。
- 索引读取必须 reinterpret 位模式，不能按 float 数值读。
- 跨块合并必须严格大于；输出单 int64 元素时 DataCopyPad 的 blockLen 用字节数（8）。
- sharedTmpBuffer 尺寸按官方"多轮累加 + 32B 对齐"要求给足，不要只给一轮。

## 验证信号

本结论来自 bench_no_harness_arg_max 首轮迭代的开发与构建：Float-AR 路径完整实现后（含 NumericLimits 编译修复），执行 `bash build.sh --soc=ascend950 -j8`。均为工具实测：

- 宿主侧三个目标编译通过、kernel 二进制生成成功：`build/op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_arg_max/` 下 5 个 `BenchNoHarnessArgMax_*.o`，并产出 `custom_opp_ubuntu_x86_64.run` 安装包（来源平台：950；证据等级：tool_observed）。
- 工程侧配套信号：op_host UT 中 InferShape 15/15 通过（含 0-dim 拒绝、dim 越界拒绝、负 dim 归一化）（来源平台：950 宿主侧；证据等级：tool_observed）。
- ST 侧为 Mock 模式 CPU golden 自测（16/16）+ fp32 基础用例 14/14 bitwise-equal——只验证测试逻辑与 golden 语义，不构成对 NPU kernel 的实测（来源平台：宿主 CPU golden；证据等级：tool_observed）。
未上板：Ascend950PR 上的运行期精度未验证。

## 失效边界

- R 超出 uint32 索引范围、或 fp16 输入 R>65535 时直接套用 half 版本会索引截断——fp16 必须先 Cast→float（未在本轨迹实现，属设计结论）。
- ARA（非最内轴）与 int32/int64/bf16 路径在本轮为 stub，本卡配方不适用；ARA 向量化路线的 padding 修补已被实测证伪，见 [ARA 向量化 padding 三方案无效卡](../precision/ara_vectorized_padding_fixes_ineffective.md)（同一算子后续联调实测）。
- 上板正确性、与 torch.argmax 的 NaN 语义对齐（NaN 参与比较时的行为）未验证。
- 其他芯片/架构未验证（未验证，推断 A3 系列 DAV_3510 同代可迁移）。

**未决项**：

- Float-AR 上板精度验证（与 torch.argmax 全量对齐，含 NaN/±0/inf 用例）未执行，是下一迭代的优先项。
- fp16/bf16 → float 前置 Cast 的 AR 路径、ARA 模式、int32/int64 路径均未实现（设计已定）。
- 双缓冲（bufNum=2）对 AR 路径的收益未评估。
## 坏实践

- 用 `>=` 做跨块合并——首最大值语义被破坏，多最大值用例必挂 bitwise。
- 直接 `SetValue(0, globalIdx)` 后不做 PipeBarrier 就 Cast——vector 结果未同步即被消费。
- 把 DataCopyPad 的 blockLen 写成元素个数（1）而不是字节数（8）——搬运长度不足。

## 误诊

无。
