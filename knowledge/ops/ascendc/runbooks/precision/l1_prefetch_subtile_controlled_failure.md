---
type: Runbook
title: "L1 预取分片须用非退化输入单独验算"
description: "同一 Cube 内核在预取深度一时精确通过，深度二失败；先恢复分支控制，再定位子 tile 搬运。"
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: "2026-10-18T08:29:56Z"
updated_at: '2026-10-20T12:33:33Z'
platforms: [a2]
---

# L1 预取分片须用非退化输入单独验算

## 现象

为验证 MatmulTransA 手工 Cube 内核的 `i>0` 子 tile 分支，隔离复制已归档的完整 `matmul.h` 与搬运助手，只在 Host 绑定增加预取深度和 baseN 的受限参数。相同稠密、有正负号、可精确二进制表示的输入上，预取深度一的所有控制均逐元素精确通过；深度二全部出现明显数值错误。没有应用修补，也没有把错误归为环境故障。

## 适用场景

工程形态为 Cube kernel 直调；API 路径为 ND→NZ `DataCopy`、dtype 分支 `LoadData`/`LoadDataWithTranspose`、裸 `Mmad` 和 Fixpipe；实际平台为 a2/Ascend910B4、CANN 9.1.0。适用粒度 `exact_task`，路线优先级 `recommended_fastpath` 指优先执行分支控制诊断。其他布局/内核不据此推出通用偏移公式。

## 系统信息

本次独立探针使用 Ascend910B4（a2）、CANN 9.1.0、PyTorch/torch_npu 2.9.0 和 Python 3.11。编译包路径为 `/usr/local/Ascend/cann-9.1.0`，以项目现有 `build_ascendc.py` 的隔离拷贝生成 CMake 并限并发构建。实际模块路径、源码/模块 SHA-256、输入输出快照和原始日志保留在审核材料；这是单次独立实验，不继承历史任务成功。

## 判别方法

对同一源码构建和输入，单独切换预取深度；同时保留最浅深度控制。输入使用小幅稠密二进制分数，参考为 CPU fp64 matmul 后回退输出 dtype，避免全一/单热点掩盖片段配对错误。原始实现和完整运行命令如下：

### matmul_prefetch/kernel/matmul.h

```cpp
/**
 * @file matmul.h
 *
 * Templated cube-only GEMM kernel used by MatmulTransA.
 *
 * Logical contract:
 *   C[M, N] = A[M, K] @ B[K, N]
 *
 * Caller is responsible for transposing the original A, padding to
 * multiples of `baseM/baseN/baseK`, and dispatching the right launcher
 * for each input dtype.
 *
 * `aType` is the dtype of A and B in GM. `outType` is the dtype of C
 * in GM. The cube always accumulates in `float`; if `outType` differs
 * from `float` the Fixpipe `quantPre` argument carries the down-cast.
 */
#ifndef MATMUL_CUSTOM_H
#define MATMUL_CUSTOM_H

#include "kernel_operator.h"
#include "matmul_tile.h"
#include "matmul_trans_a_tiling.h"

template <typename aType, typename outType>
class MatmulKernel {
    using accType = float;
    static constexpr uint32_t C0 = 32 / sizeof(aType);

public:
    __aicore__ inline MatmulKernel() {}
    __aicore__ inline void Init(uint32_t k,
                                uint32_t lda, uint32_t ldb,
                                uint32_t baseM, uint32_t baseN, uint32_t baseK,
                                uint32_t l1Prefetch,
                                AscendC::TPipe &pipe);

    __aicore__ inline void ComputeBlock(const AscendC::GlobalTensor<aType> &aBlock,
                                        const AscendC::GlobalTensor<aType> &bBlock,
                                        const AscendC::GlobalTensor<outType> &cBlock,
                                        uint32_t ldc,
                                        QuantMode_t quantPre);

private:
    __aicore__ inline void CopyA(const AscendC::GlobalTensor<aType> &A, uint32_t kLen);
    __aicore__ inline void CopyB(const AscendC::GlobalTensor<aType> &B, uint32_t kLen);
    __aicore__ inline void SplitA(const AscendC::LocalTensor<aType> &a1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void SplitB(const AscendC::LocalTensor<aType> &b1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void Compute(const AscendC::LocalTensor<accType> &c1Local,
                                   bool cmatrixInitVal);
    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<outType> &C,
                                   uint32_t ldc,
                                   QuantMode_t quantPre);

private:
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    // CO1 使用双缓冲: 当前 tile 经 Fixpipe 写回 GM 时, 下一个 tile 的 Mmad
    // 可写入另一块 L0C, 避免 L0C read/write conflict 导致的零星精度错误。
    AscendC::TQue<AscendC::TPosition::CO1, 2> outQueueCO1;

    uint32_t l1Prefetch_;
    uint32_t baseM_, baseN_, baseK_;
    uint32_t baseMK_, baseKN_, baseMN_;
    uint32_t k_;
    uint32_t aDValue_, bDValue_;
};

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::Init(
    uint32_t k, uint32_t lda, uint32_t ldb,
    uint32_t baseM, uint32_t baseN, uint32_t baseK,
    uint32_t l1Prefetch, AscendC::TPipe &pipe)
{
    ASSERT(k % baseK == 0);

    l1Prefetch_ = l1Prefetch;
    baseM_ = baseM;
    baseN_ = baseN;
    baseK_ = baseK;
    baseMK_ = baseM_ * baseK_;
    baseKN_ = baseK_ * baseN_;
    baseMN_ = baseM_ * baseN_;
    k_ = k;
    aDValue_ = lda;
    bDValue_ = ldb;

    pipe.InitBuffer(inQueueA1, 2, baseMK_ * l1Prefetch_ * sizeof(aType));
    pipe.InitBuffer(inQueueA2, 2, baseMK_ * sizeof(aType));
    pipe.InitBuffer(inQueueB1, 2, baseKN_ * l1Prefetch_ * sizeof(aType));
    pipe.InitBuffer(inQueueB2, 2, baseKN_ * sizeof(aType));
    pipe.InitBuffer(outQueueCO1, 2, baseMN_ * sizeof(accType));
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::ComputeBlock(
    const AscendC::GlobalTensor<aType> &aBlock,
    const AscendC::GlobalTensor<aType> &bBlock,
    const AscendC::GlobalTensor<outType> &cBlock,
    uint32_t ldc,
    QuantMode_t quantPre)
{
    AscendC::LocalTensor<accType> c1Local = outQueueCO1.AllocTensor<accType>();
    uint32_t kTiles = k_ / baseK_;

    for (uint32_t outer = 0; outer < kTiles; outer += l1Prefetch_) {
        uint32_t count = (kTiles - outer < l1Prefetch_) ? (kTiles - outer) : l1Prefetch_;
        uint32_t kLen = count * baseK_;

        CopyA(aBlock[outer * baseK_], kLen);
        CopyB(bBlock[outer * baseK_ * bDValue_], kLen);

        AscendC::LocalTensor<aType> a1Local = inQueueA1.DeQue<aType>();
        AscendC::LocalTensor<aType> b1Local = inQueueB1.DeQue<aType>();

        for (uint32_t i = 0; i < count; i++) {
            SplitA(a1Local, i * baseMK_, baseM_);
            SplitB(b1Local, i * baseKN_, kLen);
            Compute(c1Local, (outer + i == 0));
        }

        inQueueA1.FreeTensor(a1Local);
        inQueueB1.FreeTensor(b1Local);
    }

    outQueueCO1.EnQue(c1Local);
    CopyOut(cBlock, ldc, quantPre);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::CopyA(
    const AscendC::GlobalTensor<aType> &A, uint32_t kLen)
{
    AscendC::LocalTensor<aType> a1Local = inQueueA1.AllocTensor<aType>();
    LoadNdGmToNzL1(a1Local, A, baseM_, kLen, aDValue_);
    inQueueA1.EnQue(a1Local);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::CopyB(
    const AscendC::GlobalTensor<aType> &B, uint32_t kLen)
{
    AscendC::LocalTensor<aType> b1Local = inQueueB1.AllocTensor<aType>();
    LoadNdGmToNzL1(b1Local, B, kLen, baseN_, bDValue_);
    inQueueB1.EnQue(b1Local);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::SplitA(
    const AscendC::LocalTensor<aType> &a1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.AllocTensor<aType>();
    LoadNzL1ToZzL0A(a2Local, a1Local[offset], baseM_, baseK_, colC0Stride);
    inQueueA2.EnQue(a2Local);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::SplitB(
    const AscendC::LocalTensor<aType> &b1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<aType> b2Local = inQueueB2.AllocTensor<aType>();
    LoadNzL1ToZnL0B(b2Local, b1Local[offset], baseK_, baseN_, colC0Stride);
    inQueueB2.EnQue(b2Local);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::Compute(
    const AscendC::LocalTensor<accType> &c1Local, bool cmatrixInitVal)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.DeQue<aType>();
    AscendC::LocalTensor<aType> b2Local = inQueueB2.DeQue<aType>();
    AscendC::MmadParams mmadParams;
    mmadParams.m = baseM_;
    mmadParams.n = baseN_;
    mmadParams.k = baseK_;
    mmadParams.cmatrixInitVal = cmatrixInitVal;
    AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
    inQueueA2.FreeTensor(a2Local);
    inQueueB2.FreeTensor(b2Local);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::CopyOut(
    const AscendC::GlobalTensor<outType> &C, uint32_t ldc,
    QuantMode_t quantPre)
{
    auto c1Local = outQueueCO1.DeQue<accType>();
    FixpipeNzL0cToNdGm(C, c1Local, baseM_, baseN_, ldc, quantPre);
    // 等待 Fixpipe (PIPE_FIX) 完成后再把 CO1 buffer 归还队列,
    // 与 CO1 双缓冲共同消除 Mmad/Fixpipe 对同一块 L0C 的读写竞争。
    AscendC::PipeBarrier<PIPE_FIX>();
    outQueueCO1.FreeTensor(c1Local);
}

#endif // MATMUL_CUSTOM_H
```

### matmul_prefetch/kernel/matmul_tile.h

```cpp
/**
 * @file matmul_tile.h
 *
 * Tile-level data movement helpers for the MatmulTransA cube kernel.
 *
 * The L1->L0A/L0B path is split by dtype:
 *   - float uses LoadData3DParamsV2 (single-fractal 4B layout).
 *   - half / bfloat16 use the canonical LoadData2D / LoadDataWithTranspose
 *     paths recommended for 2B fractal matmul, avoiding layout mismatch.
 * Fixpipe always writes CO1 (float) back to GM with optional quantPre cast.
 */
#ifndef MATMUL_TILE_H
#define MATMUL_TILE_H

#include "kernel_operator.h"

// GM(ND) -> L1(Nz)  : load an (m, n) ND tile into an (m, n) Nz L1 buffer.
// Pre-condition: m % 16 == 0 && n % C0 == 0.
template<typename T>
__aicore__ inline void LoadNdGmToNzL1(const AscendC::LocalTensor<T> &dst,
                                      const AscendC::GlobalTensor<T> &src,
                                      uint32_t m, uint32_t n, uint32_t ld)
{
    AscendC::Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = m;
    params.dValue = n;
    params.srcNdMatrixStride = 0;
    params.srcDValue = ld;
    params.dstNzC0Stride = m;
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    AscendC::DataCopy(dst, src, params);
}

// L1(Nz) -> L0A(Zz)
// Generic across element widths because LoadData3DParamsV2 expresses
// extents in *elements*, not bytes.
template<typename T>
__aicore__ inline void LoadNzL1ToZzL0A(const AscendC::LocalTensor<T> &dst,
                                       const AscendC::LocalTensor<T> &src,
                                       uint32_t m, uint32_t k, uint32_t colC0Stride)
{
    AscendC::LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = k;
    params.kExtension = k;
    params.mExtension = m;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    AscendC::LoadData(dst, src, params);
}

// L1(Nz) -> L0B(Zn) — generic version that works for half / bfloat16 / float.
template<typename T>
__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<T> &dst,
                                       const AscendC::LocalTensor<T> &src,
                                       uint32_t k, uint32_t n, uint32_t colC0Stride)
{
    AscendC::LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = n;
    params.kExtension = n;
    params.mExtension = k;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    AscendC::LoadData(dst, src, params);
}

// L1(Nz) -> L0A(Zz) for half / bfloat16: canonical LoadData2D path.
// float 仍使用上方 LoadData3DParamsV2 路径。
__aicore__ inline void LoadNzL1ToZzL0A(const AscendC::LocalTensor<half> &dst,
                                       const AscendC::LocalTensor<half> &src,
                                       uint32_t m, uint32_t k, uint32_t /*colC0Stride*/)
{
    AscendC::LoadData2DParams params;
    params.repeatTimes = m * k * sizeof(half) / 512;
    params.srcStride = 1;
    params.ifTranspose = false;
    AscendC::LoadData(dst, src, params);
}

__aicore__ inline void LoadNzL1ToZzL0A(const AscendC::LocalTensor<bfloat16_t> &dst,
                                       const AscendC::LocalTensor<bfloat16_t> &src,
                                       uint32_t m, uint32_t k, uint32_t /*colC0Stride*/)
{
    AscendC::LoadData2DParams params;
    params.repeatTimes = m * k * sizeof(bfloat16_t) / 512;
    params.srcStride = 1;
    params.ifTranspose = false;
    AscendC::LoadData(dst, src, params);
}

// L1(Nz) -> L0B(Zn) — half / bfloat16 专用 LoadDataWithTranspose 路径。
__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<half> &dst,
                                       const AscendC::LocalTensor<half> &src,
                                       uint32_t k, uint32_t n, uint32_t /*colC0Stride*/)
{
    constexpr uint32_t BLOCK_SIZE = 16;
    AscendC::LoadData2dTransposeParams params;
    params.startIndex = 0;
    params.repeatTimes = k / BLOCK_SIZE;
    params.srcStride = 1;
    params.dstGap = 1;
    params.dstFracGap = 0;

    uint32_t nBlocks = n / BLOCK_SIZE;
    for (uint32_t i = 0; i < nBlocks; ++i) {
        AscendC::LoadDataWithTranspose(dst[i * BLOCK_SIZE * BLOCK_SIZE],
                                       src[i * k * BLOCK_SIZE], params);
    }
}

__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<bfloat16_t> &dst,
                                       const AscendC::LocalTensor<bfloat16_t> &src,
                                       uint32_t k, uint32_t n, uint32_t /*colC0Stride*/)
{
    constexpr uint32_t BLOCK_SIZE = 16;
    AscendC::LoadData2dTransposeParams params;
    params.startIndex = 0;
    params.repeatTimes = k / BLOCK_SIZE;
    params.srcStride = 1;
    params.dstGap = 1;
    params.dstFracGap = 0;

    uint32_t nBlocks = n / BLOCK_SIZE;
    for (uint32_t i = 0; i < nBlocks; ++i) {
        AscendC::LoadDataWithTranspose(dst[i * BLOCK_SIZE * BLOCK_SIZE],
                                       src[i * k * BLOCK_SIZE], params);
    }
}

// L0C(Nz) -> GM(ND): write an (m, n) Nz fragment back to an ND row of a
// larger GM matrix whose row stride is `dstStride` (in elements).
// `quantPre` carries any optional CO1 -> dst dtype cast (NoQuant for
// float, F322F16 for half, F322BF16 for bfloat16_t).
//
// Note: `QuantMode_t` is a *global* enum declared by the Bisheng cce
// intrinsics header, not a member of namespace AscendC.
template<typename T>
__aicore__ inline void FixpipeNzL0cToNdGm(const AscendC::GlobalTensor<T> &dst,
                                          const AscendC::LocalTensor<float> &src,
                                          uint32_t m, uint32_t n, uint32_t dstStride,
                                          QuantMode_t quantPre)
{
    AscendC::FixpipeParamsV220 params;
    params.nSize = n;
    params.mSize = m;
    params.srcStride = m;
    params.dstStride = dstStride;
    params.ndNum = 1;
    params.srcNdStride = 0;
    params.dstNdStride = 0;
    params.quantPre = quantPre;
    AscendC::Fixpipe(dst, src, params);
}

#endif // MATMUL_TILE_H
```

### matmul_prefetch/kernel/matmul_trans_a.h

```cpp
/**
 * @file matmul_trans_a.h
 *
 * Persistent cube wrapper around `MatmulKernel` for the MatmulTransA op.
 *
 * Layout:
 *   - host already supplied A.T (shape M x K) and B (shape K x N) padded
 *     to multiples of `baseM/baseN/baseK`, where M=padded_n, N=padded_k,
 *     K=padded_m.
 *   - the cube schedules (M / baseM) x (N / baseN) baseM x baseN output
 *     tiles round-robin across `usedCoreNum` AIC cores; AIV is idle (we
 *     still ask for KERNEL_TYPE_MIX_AIC_1_2 so the runtime accepts the
 *     launch on a Mix-architecture device).
 *   - each tile is computed end-to-end on one AIC; Fixpipe writes the
 *     baseM x baseN result straight to GM at `(mIdx*baseM, nIdx*baseN)`,
 *     casting from float accumulator to `outType` via `quantPre`.
 */
#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "kernel_common.h"
#include "matmul_trans_a_tiling.h"
#include "matmul.h"

template <typename aType, typename outType>
class MatmulTransAKernel {
public:
    __aicore__ inline MatmulTransAKernel() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGM,
                                AscendC::TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline QuantMode_t SelectQuantPre();

    MatmulKernel<aType, outType> mm_;
    AscendC::GlobalTensor<aType>   aGM_;
    AscendC::GlobalTensor<aType>   bGM_;
    AscendC::GlobalTensor<outType> cGM_;
    MatmulTransATiling tiling_;
    BlockScheduler sched_;
};

template <typename aType, typename outType>
__aicore__ inline QuantMode_t
MatmulTransAKernel<aType, outType>::SelectQuantPre()
{
    if constexpr (std::is_same<outType, half>::value) {
        return QuantMode_t::F322F16;
    } else if constexpr (std::is_same<outType, bfloat16_t>::value) {
        return QuantMode_t::F322BF16;
    } else {
        // float -> float, no cast
        return QuantMode_t::NoQuant;
    }
}

template <typename aType, typename outType>
__aicore__ inline void MatmulTransAKernel<aType, outType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGM, AscendC::TPipe *pipe)
{
    CopyTiling(&tiling_, tilingGM);

    aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a),
                         static_cast<uint64_t>(tiling_.M) * tiling_.K);
    bGM_.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(b),
                         static_cast<uint64_t>(tiling_.K) * tiling_.N);
    cGM_.SetGlobalBuffer(reinterpret_cast<__gm__ outType *>(c),
                         static_cast<uint64_t>(tiling_.M) * tiling_.N);

    int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    sched_.Init(tiling_.M, tiling_.N, tiling_.baseM, tiling_.baseN,
                AscendC::GetBlockNum(), coreIdx);

    if ASCEND_IS_AIC {
        mm_.Init(tiling_.K,            // reduction dim
                 tiling_.K,            // lda  (A row stride = K)
                 tiling_.N,            // ldb  (B row stride = N)
                 tiling_.baseM, tiling_.baseN, tiling_.baseK,
                 tiling_.l1Prefetch,
                 *pipe);
    }
}

template <typename aType, typename outType>
__aicore__ inline void MatmulTransAKernel<aType, outType>::Process()
{
    if ASCEND_IS_AIV {
        return; // pure cube kernel
    }

    QuantMode_t quantPre = SelectQuantPre();

    int mIdx, nIdx;
    while (sched_.HasNext()) {
        sched_.Next(mIdx, nIdx);
        auto aBlock = aGM_[mIdx * tiling_.baseM * tiling_.K];
        auto bBlock = bGM_[nIdx * tiling_.baseN];
        auto cBlock = cGM_[mIdx * tiling_.baseM * tiling_.N + nIdx * tiling_.baseN];
        mm_.ComputeBlock(aBlock, bBlock, cBlock, tiling_.N, quantPre);
    }
}
```

### matmul_prefetch/kernel/pybind11.cpp

以下实际Host完整函数绑定 `l1Prefetch` 和 `baseN`，确保paired control确实改变传入tiling参数；代码逐字保留。

```cpp
/**
 * @file pybind11.cpp
 *
 * Host entry point for the MatmulTransA AscendC operator.
 *
 * Logical op:    C = A.T @ B,  with A:(m,n), B:(m,k), C:(n,k)
 *
 * Layout strategy:
 *   - The Python wrapper is responsible for transposing A and zero-padding
 *     A.T and B so that all of (n, m, k) become multiples of 16. This pybind
 *     therefore receives `A_T` (shape padded_n x padded_m) and `B` (shape
 *     padded_m x padded_k), both already padded.
 *   - Internally we relabel (M = padded_n, K = padded_m, N = padded_k) and
 *     dispatch a vanilla GEMM cube launcher per dtype.
 *   - Output of this pybind call is the padded `C` of shape (padded_n, padded_k);
 *     the Python wrapper slices it back to (n, k) when needed.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"

#include "matmul_trans_a_tiling.h"

extern "C" void matmul_trans_a_do_fp16(uint32_t blockDim, void *stream,
                                       uint8_t *a, uint8_t *b, uint8_t *c,
                                       uint8_t *workspace, uint8_t *tiling);

extern "C" void matmul_trans_a_do_bf16(uint32_t blockDim, void *stream,
                                       uint8_t *a, uint8_t *b, uint8_t *c,
                                       uint8_t *workspace, uint8_t *tiling);

extern "C" void matmul_trans_a_do_fp32(uint32_t blockDim, void *stream,
                                       uint8_t *a, uint8_t *b, uint8_t *c,
                                       uint8_t *workspace, uint8_t *tiling);

namespace my_matmul_trans_a {

using LaunchFn = void(*)(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*);

at::Tensor run_matmul_trans_a(const at::Tensor &a_t, const at::Tensor &b, int32_t probePrefetch, int32_t probeBaseN)
{
    TORCH_CHECK(a_t.dim() == 2, "matmul_trans_a: A_T must be 2D");
    TORCH_CHECK(b.dim() == 2,   "matmul_trans_a: B must be 2D");
    TORCH_CHECK(a_t.scalar_type() == b.scalar_type(),
                "matmul_trans_a: A_T and B must share dtype");
    TORCH_CHECK(a_t.sizes()[1] == b.sizes()[0],
                "matmul_trans_a: A_T.shape[1] (padded_m) must equal B.shape[0]");

    TORCH_CHECK((probePrefetch==1||probePrefetch==2)&&(probeBaseN==16||probeBaseN==32), "bounded probe options only");
    int32_t M = static_cast<int32_t>(a_t.sizes()[0]); // padded_n
    int32_t K = static_cast<int32_t>(a_t.sizes()[1]); // padded_m
    int32_t N = static_cast<int32_t>(b.sizes()[1]);   // padded_k

    TORCH_CHECK(M % DEFAULT_BASE_M == 0, "matmul_trans_a: padded_n must be a multiple of baseM");
    TORCH_CHECK(N % DEFAULT_BASE_N == 0, "matmul_trans_a: padded_k must be a multiple of baseN");
    TORCH_CHECK(K % DEFAULT_BASE_K == 0, "matmul_trans_a: padded_m must be a multiple of baseK");

    LaunchFn launchFn = nullptr;
    at::ScalarType outDtype = a_t.scalar_type();

    if (a_t.scalar_type() == at::kHalf) {
        launchFn = matmul_trans_a_do_fp16;
    } else if (a_t.scalar_type() == at::kBFloat16) {
        launchFn = matmul_trans_a_do_bf16;
    } else if (a_t.scalar_type() == at::kFloat) {
        launchFn = matmul_trans_a_do_fp32;
    } else {
        TORCH_CHECK(false, "matmul_trans_a: unsupported dtype, expected float16/bfloat16/float32");
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    // 20 AIC cores match the design's persistent schedule. The BlockScheduler
    // distributes (M/baseM)*(N/baseN) tiles round-robin so it is safe to run
    // even when there are more cores than tiles (extra cores get zero work).
    uint32_t usedCoreNum = 20;

    at::Tensor c = at::empty({M, N}, at::device(at::kPrivateUse1).dtype(outDtype));

    // Tiling struct (CPU side, then pushed to device)
    at::Tensor t = at::empty({(int64_t)sizeof(MatmulTransATiling)},
                              at::device(at::kCPU).dtype(at::kByte));
    auto *tp = reinterpret_cast<MatmulTransATiling *>(t.data_ptr());
    tp->M          = M;
    tp->N          = N;
    tp->K          = K;
    tp->baseM      = DEFAULT_BASE_M;
    tp->baseN      = probeBaseN;
    tp->baseK      = DEFAULT_BASE_K;
    tp->l1Prefetch = probePrefetch;
    auto tiling_npu = t.to(at::kPrivateUse1);

    // Workspace is unused by this pure-cube kernel but the launcher
    // signature still expects an address; allocate a tiny stub.
    at::Tensor w = at::empty({64}, at::device(at::kPrivateUse1).dtype(at::kByte));

    launchFn(usedCoreNum, acl_stream,
             static_cast<uint8_t*>(const_cast<void*>(a_t.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(b.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(c.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(w.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(tiling_npu.storage().data())));
    return c;
}

} // namespace my_matmul_trans_a

PYBIND11_MODULE(_prefetch_a_probe, m)
{
    m.doc() = "matmul_trans_a pybind11 interfaces";
    m.def("run_matmul_trans_a", &my_matmul_trans_a::run_matmul_trans_a, "");
}
```

### run_prefetch.py

```python
import hashlib,json,sys,os
from pathlib import Path
import torch,torch_npu
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'matmul_prefetch/kernel/build'))
import _prefetch_a_probe as ext

def h(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
path=ROOT/'npu_prefetch';path.mkdir(exist_ok=False)
torch.npu.set_device(0);records=[];tensors={}
for dtype in [torch.float16,torch.float32]:
 for k in [32,64]:
  m,n=16,32
  # Exact binary fractions and small products make CPU fp64 reference unambiguous.
  a=((torch.arange(m*k).reshape(m,k)*7+3)%17-8).to(dtype)/16
  b=((torch.arange(k*n).reshape(k,n)*11+1)%19-9).to(dtype)/16
  ref=(a.double()@b.double()).to(dtype)
  for prefetch in [1,2]:
   for base_n in [16,32]:
    y=ext.run_matmul_trans_a(a.to('npu:0'),b.to('npu:0'),prefetch,base_n)
    torch.npu.synchronize();y=y.cpu();prefix=f'{dtype}_{k}_{prefetch}_{base_n}'
    tensors[prefix+'_a']=a;tensors[prefix+'_b']=b;tensors[prefix+'_ref']=ref;tensors[prefix+'_out']=y
    records.append({'dtype':str(dtype),'M':m,'N':n,'K':k,'l1Prefetch':prefetch,'baseN':base_n,'equal_elements':int((y==ref).sum()),'total':y.numel(),'max_abs_diff':float((y.double()-ref.double()).abs().max()),'a_sha256':h(a),'b_sha256':h(b),'out_sha256':h(y),'reference_sha256':h(ref)})
torch.save(tensors,path/'inputs_outputs.pt')
result={'schema_version':'expanded_local.group_a.prefetch.v1','device':torch.npu.get_device_name(0),'logical_device':0,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'reference':'CPU fp64 matmul then cast; small exact binary fractions','module_path':ext.__file__,'module_sha256':hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'matmul_prefetch/kernel').iterdir()) if p.is_file()},'records':records,'limitations':['Independent new experiment, historical task outcome not recertified','Changes only host runtime prefetch/baseN; no fix applied','A failure does not by itself isolate base offset vs per-column helper stride or API path','No official evaluation or latency claims']}
(path/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
```

## 根因

已经定位到启用非首个 L1 子 tile 的路径，但没有追到唯一机制。`i*baseKN`、列分形 stride 与 dtype 专用搬运助手均为待区分解释，任何一个静态模型都不能替代设备修复验收。

## 验证信号

来源为本次隔离复制内核后的实际加载模块和冻结 CPU/NPU 输入输出。

- 预取深度一的八组控制各为 512/512 精确匹配，最大绝对误差为零。（来源平台：a2；证据等级：artifact_verified）
- 相同八组输入切深度二后全部失败，每组只有 1–6 个输出元素与参考相等。（来源平台：a2；证据等级：artifact_verified）

## 设计决策

先证明默认分支在本环境正确，再激活 `i>0`。不以旧默认用例通过推断该分支覆盖，也不直接照搬源码注释的通用偏移断言。

## 坏实践

默认预取深度只执行 `i=0` 时，以整算子通过证明非首子 tile 的偏移正确。

## 性能影响

未测性能，无更深预取的速度收益声明。

## 失效边界

- 优先项：复现并追根因到机制层，分别区分子 tile 基址、列 stride、dtype 搬运助手，并用现有 Debug Agent 执行后续合法修复。
- 当前仅为失败诊断，不提供已验证修法。任何新修法须回放本次冻结输入并覆盖默认正确路径。

## 预防

测试必须显式激活非首 tile 分支，并检查完整稠密输出，不能只测输入幅值或收缩计数。

## 误诊

静态 NZ 基址反例只是线索；它既不能证明原来的基址在所有布局上错，也不能认证替代基址普遍正确。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/evidence/prefetch.json#/records`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/evidence/prefetch_bits.json`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/matmul_prefetch/kernel/matmul.h`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/matmul_prefetch/kernel/matmul_tile.h`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/matmul_prefetch/kernel/matmul_trans_a.h`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/matmul_prefetch/kernel/pybind11.cpp`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/a/run_prefetch.py`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。 公开dynamic投影保留数值、设备名、源码/模块或脚本身份及9.1.0版本标记；不包含完整同PID runtime库路径/hash，不能据此独立认证全部加载库。
